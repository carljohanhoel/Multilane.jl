const GLOBAL_DEBUG = 1

import Gallium

# macro if_debug(expr)
#     return quote
#         if GLOBAL_DEBUG > 0 # && isinteractive()
#             $(esc(expr))
#         end
#     end
# end

macro if_debug(expr) end


# To debug no_crash_model:
# Uncomment longer macro expression
# Comment last line
# Run sim
# Variables are saved at crash
