from ngsxditto.fluid import FluidParameters, WallParameters

def test_fluidparams():
    nu = 1.1
    rho = 1.2
    stc = 1.3
    params = FluidParameters(viscosity=nu, density=rho, surface_tension_coeff=stc) 
    assert params["viscosity"] == nu
    assert params["density"] == rho
    assert params["surface_tension_coeff"] == stc

def test_update_fluidparams():
    nu = 1.1
    rho = 1.2
    stc = 1.3
    params = FluidParameters(viscosity=nu, density=rho, surface_tension_coeff=stc) 
    for param in [nu, rho, stc]:
        param -= 1
    params.Update(viscosity=nu, density=rho, surface_tension_coeff=stc)
    assert params["viscosity"] == nu
    assert params["density"] == rho
    assert params["surface_tension_coeff"] == stc


def test_wallparams():
    nu = 1.1
    rho = 1.2
    params = WallParameters(friction_coeff=nu, contact_angle=rho) 
    assert params["friction_coeff"] == nu
    assert params["contact_angle"] == rho

def test_update_wallparams():
    nu = 1.1
    rho = 1.2
    params = WallParameters(friction_coeff=nu, contact_angle=rho) 
    for param in [nu, rho]:
        param -= 1
    params.Update(friction_coeff=nu, contact_angle=rho)
    assert params["friction_coeff"] == nu
    assert params["contact_angle"] == rho

if __name__=="__main__":
    test_params()