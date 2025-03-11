@[simp]
lemma LinearMap.charpoly_baseChange {R M} [CommRing R] [AddCommGroup M] [Module R M]
    [Module.Free R M] [Module.Finite R M] (f : M →ₗ[R] M)
    (A) [CommRing A] [Algebra R A] :
    (f.baseChange A).charpoly = f.charpoly.map (algebraMap R A) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq (LinearMap.baseChange A f).charpoly (Polynomial.map (algebraMap R A) f.ch …
  -/
  nontriviality A
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a✝ : Nontrivial A
    ⊢ Eq (LinearMap.baseChange A f).charpoly (Polynomial.map (algebraMap R A) f.ch …
  -/
  have := (algebraMap R A).domain_nontrivial
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a✝ : Nontrivial A
    this : Nontrivial R
    ⊢ Eq (LinearMap.baseChange A f).charpoly (Polynomial.map (algebraMap R A) f.ch …
  -/
  let I := Module.Free.ChooseBasisIndex R M
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a✝ : Nontrivial A
    this : Nontrivial R
    I : Type u_2 := Module.Free.ChooseBasisIndex R M
    ⊢ Eq (LinearMap.baseChange A f).charpoly (Polynomial.map (algebraMap R A) f.ch …
  -/
  let b : Basis I R M := Module.Free.chooseBasis R M
  rw [← f.charpoly_toMatrix b, ← (f.baseChange A).charpoly_toMatrix (b.baseChange A),
    ← Matrix.charpoly_map]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a✝ : Nontrivial A
    this : Nontrivial R
    I : Type u_2 := Module.Free.ChooseBasisIndex R M
    b : Basis I R M := Module.Free.chooseBasis R M
    ⊢ Eq ((LinearMap.toMatrix (Basis.baseChange A b) (Basis.baseChange A b)) (Line …
  -/
  congr 1
  /-
    case e_M
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a✝ : Nontrivial A
    this : Nontrivial R
    I : Type u_2 := Module.Free.ChooseBasisIndex R M
    b : Basis I R M := Module.Free.chooseBasis R M
    ⊢ Eq ((LinearMap.toMatrix (Basis.baseChange A b) (Basis.baseChange A b)) (Line …
  -/
  ext i j
  /-
    case e_M.a
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a✝ : Nontrivial A
    this : Nontrivial R
    I : Type u_2 := Module.Free.ChooseBasisIndex R M
    b : Basis I R M := Module.Free.chooseBasis R M
    i j : I
    ⊢ Eq ((LinearMap.toMatrix (Basis.baseChange A b) (Basis.baseChange A b)) (Line …
  -/
  simp [LinearMap.toMatrix_apply, ← Algebra.algebraMap_eq_smul_one]
  /-
    🎉 no goals
  -/


lemma LinearMap.det_eq_sign_charpoly_coeff {R M} [CommRing R] [AddCommGroup M] [Module R M]
    [Module.Free R M] [Module.Finite R M] (f : M →ₗ[R] M) :
    LinearMap.det f = (-1) ^ Module.finrank R M * f.charpoly.coeff 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq (LinearMap.det f) (HMul.hMul (HPow.hPow (-1) (Module.finrank R M)) (f.cha …
  -/
  nontriviality R
  rw [← LinearMap.det_toMatrix (Module.Free.chooseBasis R M), Matrix.det_eq_sign_charpoly_coeff,
    ← Module.finrank_eq_card_chooseBasisIndex, charpoly_def]


lemma LinearMap.det_baseChange {R M} [CommRing R] [AddCommGroup M] [Module R M]
    [Module.Free R M] [Module.Finite R M]
    {A} [CommRing A] [Algebra R A] (f : M →ₗ[R] M) :
    LinearMap.det (f.baseChange A) = algebraMap R A (LinearMap.det f) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq (LinearMap.det (LinearMap.baseChange A f)) ((algebraMap R A) (LinearMap.d …
  -/
  nontriviality A
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M M
    a✝ : Nontrivial A
    ⊢ Eq (LinearMap.det (LinearMap.baseChange A f)) ((algebraMap R A) (LinearMap.d …
  -/
  have := (algebraMap R A).domain_nontrivial
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M M
    a✝ : Nontrivial A
    this : Nontrivial R
    ⊢ Eq (LinearMap.det (LinearMap.baseChange A f)) ((algebraMap R A) (LinearMap.d …
  -/
  rw [LinearMap.det_eq_sign_charpoly_coeff, LinearMap.det_eq_sign_charpoly_coeff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : LinearMap (RingHom.id R) M M
    a✝ : Nontrivial A
    this : Nontrivial R
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) (Module.finrank A (TensorProduct R A M))) ((Li …
  -/
  simp
  /-
    🎉 no goals
  -/


