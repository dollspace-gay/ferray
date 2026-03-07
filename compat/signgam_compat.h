/*
 * Compatibility header for Windows builds.
 *
 * The CORE-MATH lgamma.c references `signgam`, a POSIX global declared in
 * <math.h> on glibc but absent from the Windows CRT.  core-math-sys ships
 * its own `int signgam;` definition in lib/signgam.c, so we only need the
 * declaration visible during compilation of lgamma.c.
 *
 * Inject via:  CFLAGS="-include <absolute-path>/signgam_compat.h"
 *
 * See: https://github.com/dollspace-gay/ferray/issues/5
 */
#if defined(_WIN32) && !defined(signgam)
extern int signgam;
#endif
