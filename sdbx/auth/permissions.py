from typing import TypedDict

import jwt


class sdbxJwt(TypedDict, total=False):
    sub: str


def jwt_decode(user_token: str) -> sdbxJwt:
    # todo: set up a way for users to override this behavior easily
    return sdbxJwt(**jwt.decode(user_token, algorithms=['HS256', "none"],
                                 # todo: this should be configurable
                                 options={"verify_signature": False, 'verify_aud': False, 'verify_iss': False}))
