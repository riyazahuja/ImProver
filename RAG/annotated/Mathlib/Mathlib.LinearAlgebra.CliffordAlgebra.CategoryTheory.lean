/-- The "clifford algebra" functor, sending a quadratic `R`-module `V` to the clifford algebra on
`V`.

This is `CliffordAlgebra.map` through the lens of category theory. -/
@[simps]
def QuadraticModuleCat.cliffordAlgebra : QuadraticModuleCat.{u} R ⥤ AlgebraCat.{u} R where
  obj M := AlgebraCat.of R (CliffordAlgebra M.form)
  map {_M _N} f := AlgebraCat.ofHom <| CliffordAlgebra.map f.toIsometry
                  /-
                    R : Type u
                    inst✝ : CommRing R
                    _M : QuadraticModuleCat R
                    ⊢ Eq ({ obj := fun M => AlgebraCat.of R (CliffordAlgebra M.form), map := fun { …
                  -/
  map_id _M := by simp
                  /-
                    🎉 no goals
                  -/
                                /-
                                  R : Type u
                                  inst✝ : CommRing R
                                  _M _N _P : QuadraticModuleCat R
                                  f : Quiver.Hom _M _N
                                  g : Quiver.Hom _N _P
                                  ⊢ Eq ({ obj := fun M => AlgebraCat.of R (CliffordAlgebra M.form), map := fun { …
                                -/
  map_comp {_M _N _P} f g := by ext; simp
                                     /-
                                       🎉 no goals
                                     -/

