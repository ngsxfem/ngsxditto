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
    mu = 1.2
    rho = 1.3
    params = WallParameters(friction_coeff_surface=nu,friction_coeff_line=mu, contact_angle=rho)
    assert params["friction_coeff_surface"] == nu
    assert params["friction_coeff_line"] == mu
    assert params["contact_angle"] == rho

def test_update_wallparams():
    nu = 1.1
    mu = 1.2
    rho = 1.3
    params = WallParameters(friction_coeff_surface=nu, friction_coeff_line=mu, contact_angle=rho)
    for param in [nu, mu, rho]:
        param -= 1
    params.Update(friction_coeff_surface=nu, friction_coeff_line=mu, contact_angle=rho)
    assert params["friction_coeff_surface"] == nu
    assert params["friction_coeff_line"] == mu
    assert params["contact_angle"] == rho
