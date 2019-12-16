import os
import bretina

curr_ver = bretina.__version__

if 'CI_COMMIT_TAG' in os.environ:
    exp_ver = os.environ['CI_COMMIT_TAG'].strip('v')

    msg = 'Version mismatch! (Current={0} != Expected={1})'.format(curr_ver, exp_ver)
    assert curr_ver == exp_ver, msg
else:
    print(f"Version {curr_ver} can not be verified since not runned from gitlab CI")
