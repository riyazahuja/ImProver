/-- If `A`, `B` are `R`-algebras, `R` injects into `A` and `B`, and `A` and `B` are domains
(which implies `R` is also a domain), then `A ⊗[R] B` is nontrivial. -/
theorem nontrivial_of_algebraMap_injective_of_isDomain
    (R A B : Type*) [CommRing R] [CommRing A] [CommRing B] [Algebra R A] [Algebra R B]
    (ha : Function.Injective (algebraMap R A)) (hb : Function.Injective (algebraMap R B))
    [IsDomain A] [IsDomain B] : Nontrivial (A ⊗[R] B) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    ⊢ Nontrivial (TensorProduct R A B)
  -/
  haveI := ha.isDomain _
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    this : IsDomain R
    ⊢ Nontrivial (TensorProduct R A B)
  -/
  let FR := FractionRing R
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    this : IsDomain R
    FR : Type u_1 := FractionRing R
    ⊢ Nontrivial (TensorProduct R A B)
  -/
  let FA := FractionRing A
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    this : IsDomain R
    FR : Type u_1 := FractionRing R
    FA : Type u_2 := FractionRing A
    ⊢ Nontrivial (TensorProduct R A B)
  -/
  let FB := FractionRing B
  let fa : FR →ₐ[R] FA := IsFractionRing.liftAlgHom (g := Algebra.ofId R FA)
    ((IsFractionRing.injective A FA).comp ha)
  let fb : FR →ₐ[R] FB := IsFractionRing.liftAlgHom (g := Algebra.ofId R FB)
    ((IsFractionRing.injective B FB).comp hb)
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    ha : Function.Injective ⇑(algebraMap R A)
    hb : Function.Injective ⇑(algebraMap R B)
    inst✝¹ : IsDomain A
    inst✝ : IsDomain B
    this : IsDomain R
    FR : Type u_1 := FractionRing R
    FA : Type u_2 := FractionRing A
    FB : Type u_3 := FractionRing B
    fa : AlgHom R FR FA := IsFractionRing.liftAlgHom ⋯
    fb : AlgHom R FR FB := IsFractionRing.liftAlgHom ⋯
    ⊢ Nontrivial (TensorProduct R A B)
  -/
  algebraize_only [fa.toRingHom, fb.toRingHom]
  exact Algebra.TensorProduct.mapOfCompatibleSMul FR R FA FB |>.comp
    (Algebra.TensorProduct.map (IsScalarTower.toAlgHom R A FA) (IsScalarTower.toAlgHom R B FB))
    |>.toRingHom.domain_nontrivial


