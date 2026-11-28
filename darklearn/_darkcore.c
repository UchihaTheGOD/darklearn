/* darklearn/_darkcore.c — DarkCore's optional C core.
 *
 * Same kernels as darkcore.py, in C: tiled matmul (float, row-major,
 * block 32, i-k-j order), fused linear+ReLU, and the fused SGD step.
 * No external libraries, only the C stdlib and the CPython C API.
 *
 * Inputs/outputs cross the boundary as raw float32 bytes: the Python
 * side passes array('f') objects (buffer protocol) and gets `bytes`
 * back, which it wraps with array('f').frombytes(). That keeps this
 * file free of any knowledge of darklearn's Python classes.
 *
 * The C core is strictly optional: if it is not built, darkcore.py
 * falls back to the pure-Python kernels with no error. Build with:
 *     python3 scripts/build_c.py
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

/* --- helpers ---------------------------------------------------------- */

static int
get_float_buffer(PyObject *obj, Py_buffer *buf, const char *name)
{
    if (PyObject_GetBuffer(obj, buf, PyBUF_SIMPLE) < 0) {
        PyErr_Format(PyExc_TypeError, "%s must support the buffer protocol", name);
        return -1;
    }
    if (buf->len % 4 != 0) {
        PyBuffer_Release(buf);
        PyErr_Format(PyExc_ValueError, "%s length is not a multiple of 4 bytes", name);
        return -1;
    }
    return 0;
}

static int
check_len(Py_buffer *buf, Py_ssize_t need, const char *name)
{
    if (buf->len < need * (Py_ssize_t)sizeof(float)) {
        PyErr_Format(PyExc_ValueError,
                     "%s too short: need %zd float32 elements, got %zd bytes",
                     name, need, buf->len);
        return -1;
    }
    return 0;
}

/* --- tiled matmul ----------------------------------------------------- */
/* matmul(a, b, m, k, n, block=32) -> bytes of shape (m, n), float32.
 * a: (m, k), b: (k, n), both row-major. i-k-j order, tiles of `block`.
 */
static PyObject *
darkcore_matmul(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj;
    Py_ssize_t m, k, n;
    long block = 32;
    if (!PyArg_ParseTuple(args, "OOnnn|l", &a_obj, &b_obj, &m, &k, &n, &block))
        return NULL;
    if (m < 0 || k < 0 || n < 0 || block <= 0) {
        PyErr_SetString(PyExc_ValueError, "m, k, n must be >= 0 and block > 0");
        return NULL;
    }

    Py_buffer a_buf, b_buf;
    if (get_float_buffer(a_obj, &a_buf, "a") < 0) return NULL;
    if (get_float_buffer(b_obj, &b_buf, "b") < 0) {
        PyBuffer_Release(&a_buf);
        return NULL;
    }
    if (check_len(&a_buf, m * k, "a") < 0 ||
        check_len(&b_buf, k * n, "b") < 0) {
        PyBuffer_Release(&a_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }

    PyObject *out = PyBytes_FromStringAndSize(NULL, m * n * (Py_ssize_t)sizeof(float));
    if (out == NULL) {
        PyBuffer_Release(&a_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }
    float *a = (float *)a_buf.buf;
    float *b = (float *)b_buf.buf;
    float *o = (float *)PyBytes_AS_STRING(out);
    memset(o, 0, (size_t)(m * n * sizeof(float)));

    for (Py_ssize_t ii = 0; ii < m; ii += block) {
        Py_ssize_t i_end = ii + block < m ? ii + block : m;
        for (Py_ssize_t kk = 0; kk < k; kk += block) {
            Py_ssize_t k_end = kk + block < k ? kk + block : k;
            for (Py_ssize_t jj = 0; jj < n; jj += block) {
                Py_ssize_t j_end = jj + block < n ? jj + block : n;
                for (Py_ssize_t i = ii; i < i_end; i++) {
                    float *o_row = o + i * n;
                    const float *a_row = a + i * k;
                    for (Py_ssize_t k2 = kk; k2 < k_end; k2++) {
                        float aik = a_row[k2];
                        const float *b_row = b + k2 * n;
                        for (Py_ssize_t j = jj; j < j_end; j++)
                            o_row[j] += aik * b_row[j];
                    }
                }
            }
        }
    }

    PyBuffer_Release(&a_buf);
    PyBuffer_Release(&b_buf);
    return out;
}

/* --- fused linear + relu ---------------------------------------------- */
/* fused_linear_relu(x, w, b, batch, in_f, out_f) -> (out_bytes, mask_bytes)
 * x: (batch, in_f), w: (out_f, in_f), b: (out_f,).
 * out: (batch, out_f) float32 bytes; mask: (batch*out_f) bytes, 1 where
 * the pre-ReLU value was > 0. One pass over x/W, no intermediates.
 */
static PyObject *
darkcore_fused_linear_relu(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *w_obj, *b_obj;
    Py_ssize_t batch, in_f, out_f;
    if (!PyArg_ParseTuple(args, "OOOnnn", &x_obj, &w_obj, &b_obj,
                          &batch, &in_f, &out_f))
        return NULL;
    if (batch < 0 || in_f < 0 || out_f < 0) {
        PyErr_SetString(PyExc_ValueError, "batch, in_f, out_f must be >= 0");
        return NULL;
    }

    Py_buffer x_buf, w_buf, b_buf;
    if (get_float_buffer(x_obj, &x_buf, "x") < 0) return NULL;
    if (get_float_buffer(w_obj, &w_buf, "w") < 0) {
        PyBuffer_Release(&x_buf);
        return NULL;
    }
    if (get_float_buffer(b_obj, &b_buf, "b") < 0) {
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&w_buf);
        return NULL;
    }
    if (check_len(&x_buf, batch * in_f, "x") < 0 ||
        check_len(&w_buf, out_f * in_f, "w") < 0 ||
        check_len(&b_buf, out_f, "b") < 0) {
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&w_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }

    PyObject *out = PyBytes_FromStringAndSize(
        NULL, batch * out_f * (Py_ssize_t)sizeof(float));
    PyObject *mask = PyBytes_FromStringAndSize(NULL, batch * out_f);
    if (out == NULL || mask == NULL) {
        Py_XDECREF(out);
        Py_XDECREF(mask);
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&w_buf);
        PyBuffer_Release(&b_buf);
        return NULL;
    }
    float *x = (float *)x_buf.buf;
    float *w = (float *)w_buf.buf;
    float *b = (float *)b_buf.buf;
    float *o = (float *)PyBytes_AS_STRING(out);
    unsigned char *mk = (unsigned char *)PyBytes_AS_STRING(mask);
    memset(o, 0, (size_t)(batch * out_f * sizeof(float)));

    for (Py_ssize_t i = 0; i < batch; i++) {
        float *o_row = o + i * out_f;
        const float *x_row = x + i * in_f;
        for (Py_ssize_t k2 = 0; k2 < in_f; k2++) {
            float xv = x_row[k2];
            for (Py_ssize_t j = 0; j < out_f; j++)
                o_row[j] += xv * w[j * in_f + k2];
        }
    }
    for (Py_ssize_t i = 0; i < batch * out_f; i++) {
        float v = o[i] + b[i % out_f];
        if (v > 0.0f) {
            o[i] = v;
            mk[i] = 1;
        } else {
            o[i] = 0.0f;
            mk[i] = 0;
        }
    }

    PyBuffer_Release(&x_buf);
    PyBuffer_Release(&w_buf);
    PyBuffer_Release(&b_buf);
    PyObject *res = PyTuple_Pack(2, out, mask);
    Py_DECREF(out);
    Py_DECREF(mask);
    return res;
}

/* --- fused SGD step -------------------------------------------------- */
/* fused_sgd(p, g, v_or_None, lr, momentum, wd) -> velocity bytes or None
 * p: writable float32 buffer (params, updated in place).
 * g: readable float32 buffer (grads, same length as p).
 * v: writable float32 buffer (velocity, same length) or None.
 * Per element, in one C pass (same maths as optim.SGD / the Python
 * fused_sgd_step_py, computed in double and rounded once per store
 * so the results match the Python loop to float32 rounding):
 *   with momentum:  gv = g[i] + wd*p[i]; v[i] = momentum*v[i] + gv;
 *                   p[i] -= lr*v[i]
 *   without:        p[i] -= lr*(g[i] + wd*p[i])
 * When momentum != 0 and no velocity is passed, a zeroed velocity is
 * created and returned as bytes (the Python side wraps it in an
 * array('f') and keeps it as the state). Otherwise returns None.
 */
static PyObject *
darkcore_fused_sgd(PyObject *self, PyObject *args)
{
    PyObject *p_obj, *g_obj, *v_obj;
    double lr, momentum, wd;
    if (!PyArg_ParseTuple(args, "OOOddd", &p_obj, &g_obj, &v_obj,
                          &lr, &momentum, &wd))
        return NULL;
    int have_v = (v_obj != Py_None);
    if (momentum == 0.0)
        have_v = 0;  /* no velocity is touched without momentum */

    Py_buffer p_buf, g_buf, v_buf;
    memset(&v_buf, 0, sizeof(v_buf));
    if (PyObject_GetBuffer(p_obj, &p_buf, PyBUF_WRITABLE) < 0)
        return NULL;
    if (PyObject_GetBuffer(g_obj, &g_buf, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&p_buf);
        return NULL;
    }
    if (have_v && PyObject_GetBuffer(v_obj, &v_buf, PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&p_buf);
        PyBuffer_Release(&g_buf);
        return NULL;
    }
    Py_ssize_t n = p_buf.len / (Py_ssize_t)sizeof(float);
    if (p_buf.len % (Py_ssize_t)sizeof(float) != 0 ||
        g_buf.len < n * (Py_ssize_t)sizeof(float) ||
        (have_v && v_buf.len < n * (Py_ssize_t)sizeof(float))) {
        PyErr_SetString(PyExc_ValueError,
                        "p, g and v (when given) must hold the same "
                        "number of float32 elements");
        PyBuffer_Release(&p_buf);
        PyBuffer_Release(&g_buf);
        if (have_v)
            PyBuffer_Release(&v_buf);
        return NULL;
    }

    float *p = (float *)p_buf.buf;
    const float *g = (const float *)g_buf.buf;
    float *v = have_v ? (float *)v_buf.buf : NULL;
    PyObject *created = NULL;
    if (momentum != 0.0 && !have_v) {
        created = PyBytes_FromStringAndSize(
            NULL, n * (Py_ssize_t)sizeof(float));
        if (created == NULL) {
            PyBuffer_Release(&p_buf);
            PyBuffer_Release(&g_buf);
            return NULL;
        }
        v = (float *)PyBytes_AS_STRING(created);
        memset(v, 0, (size_t)(n * sizeof(float)));
    }

    if (momentum != 0.0) {
        /* Same evaluation order as the Python loop
         * (mo*v + g + wd*p, left to right) so results match it to
         * float32 rounding. */
        for (Py_ssize_t i = 0; i < n; i++) {
            v[i] = (float)(momentum * (double)v[i] + (double)g[i] +
                           wd * (double)p[i]);
            p[i] = (float)((double)p[i] - lr * (double)v[i]);
        }
    } else {
        for (Py_ssize_t i = 0; i < n; i++)
            p[i] = (float)((double)p[i] -
                           lr * ((double)g[i] + wd * (double)p[i]));
    }

    PyBuffer_Release(&p_buf);
    PyBuffer_Release(&g_buf);
    if (have_v)
        PyBuffer_Release(&v_buf);
    if (created != NULL)
        return created;
    Py_RETURN_NONE;
}

/* --- module ----------------------------------------------------------- */

static PyMethodDef DarkcoreMethods[] = {
    {"matmul", darkcore_matmul, METH_VARARGS,
     "matmul(a, b, m, k, n, block=32) -> bytes: tiled float32 matmul."},
    {"fused_linear_relu", darkcore_fused_linear_relu, METH_VARARGS,
     "fused_linear_relu(x, w, b, batch, in_f, out_f) -> (bytes, bytes)."},
    {"fused_sgd", darkcore_fused_sgd, METH_VARARGS,
     "fused_sgd(p, g, v_or_None, lr, momentum, wd) -> bytes or None: "
     "in-place fused SGD step."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef darkcore_module = {
    PyModuleDef_HEAD_INIT, "_darkcore",
    "DarkCore C core: tiled matmul + fused linear+ReLU (optional).",
    -1, DarkcoreMethods,
};

PyMODINIT_FUNC
PyInit__darkcore(void)
{
    return PyModule_Create(&darkcore_module);
}
