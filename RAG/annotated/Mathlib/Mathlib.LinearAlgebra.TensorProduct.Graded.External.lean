instance (i : ι × ι) : Module R (𝒜 (Prod.fst i) ⊗[R] ℬ (Prod.snd i)) :=
  TensorProduct.leftModule


local notation "𝒜ℬ" => (fun i : ι × ι => 𝒜 (Prod.fst i) ⊗[R] ℬ (Prod.snd i))

local notation "ℬ𝒜" => (fun i : ι × ι => ℬ (Prod.fst i) ⊗[R] 𝒜 (Prod.snd i))


/-- Auxliary construction used to build `TensorProduct.gradedComm`.

This operates on direct sums of tensors instead of tensors of direct sums. -/
def gradedCommAux : DirectSum _ 𝒜ℬ →ₗ[R] DirectSum _ ℬ𝒜 := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    ⊢ LinearMap (RingHom.id R) (DirectSum (Prod ι ι) fun i => TensorProduct R (𝒜 i …
  -/
  refine DirectSum.toModule R _ _ fun i => ?_
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i : Prod ι ι
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (𝒜 i.1) (ℬ i.2)) (DirectSum (Prod  …
  -/
  have o := DirectSum.lof R _ ℬ𝒜 i.swap
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i : Prod ι ι
    o : LinearMap (RingHom.id R) (TensorProduct R (ℬ i.swap.1) (𝒜 i.swap.2)) (Dire …
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (𝒜 i.1) (ℬ i.2)) (DirectSum (Prod  …
  -/
  have s : ℤˣ := ((-1 : ℤˣ)^(i.1* i.2 : ι) : ℤˣ)
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i : Prod ι ι
    o : LinearMap (RingHom.id R) (TensorProduct R (ℬ i.swap.1) (𝒜 i.swap.2)) (Dire …
    s : Units Int
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (𝒜 i.1) (ℬ i.2)) (DirectSum (Prod  …
  -/
  exact (s • o) ∘ₗ (TensorProduct.comm R _ _).toLinearMap
  /-
    🎉 no goals
  -/


@[simp]
theorem gradedCommAux_lof_tmul (i j : ι) (a : 𝒜 i) (b : ℬ j) :
    gradedCommAux R 𝒜 ℬ (lof R _ 𝒜ℬ (i, j) (a ⊗ₜ b)) =
      (-1 : ℤˣ)^(j * i) • lof R _ ℬ𝒜 (j, i) (b ⊗ₜ a) := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i j : ι
    a : 𝒜 i
    b : ℬ j
    ⊢ Eq ((TensorProduct.gradedCommAux R 𝒜 ℬ) ((DirectSum.lof R (Prod ι ι) (fun i  …
  -/
  rw [gradedCommAux]
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i j : ι
    a : 𝒜 i
    b : ℬ j
    ⊢ Eq ((DirectSum.toModule R (Prod ι ι) (DirectSum (Prod ι ι) fun i => TensorPr …
  -/
  dsimp
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i j : ι
    a : 𝒜 i
    b : ℬ j
    ⊢ Eq ((DirectSum.toModule R (Prod ι ι) (DirectSum (Prod ι ι) fun i => TensorPr …
  -/
  simp [mul_comm i j]
  /-
    🎉 no goals
  -/


@[simp]
theorem gradedCommAux_comp_gradedCommAux :
    gradedCommAux R 𝒜 ℬ ∘ₗ gradedCommAux R ℬ 𝒜 = LinearMap.id := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    ⊢ Eq ((TensorProduct.gradedCommAux R 𝒜 ℬ).comp (TensorProduct.gradedCommAux R  …
  -/
  ext i a b
  /-
    case H.a.h.h
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i : Prod ι ι
    a : ℬ i.1
    b : 𝒜 i.2
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.gradedCommAux …
  -/
  dsimp
  rw [gradedCommAux_lof_tmul, LinearMap.map_smul_of_tower, gradedCommAux_lof_tmul, smul_smul,
    mul_comm i.2 i.1, Int.units_mul_self, one_smul]


/-- The braiding operation for tensor products of externally `ι`-graded algebras.

This sends $a ⊗ b$ to $(-1)^{\deg a' \deg b} (b ⊗ a)$. -/
def gradedComm :
    (⨁ i, 𝒜 i) ⊗[R] (⨁ i, ℬ i) ≃ₗ[R] (⨁ i, ℬ i) ⊗[R] (⨁ i, 𝒜 i) := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    ⊢ LinearEquiv (RingHom.id R) (TensorProduct R (DirectSum ι fun i => 𝒜 i) (Dire …
  -/
  refine TensorProduct.directSum R R 𝒜 ℬ ≪≫ₗ ?_ ≪≫ₗ (TensorProduct.directSum R R ℬ 𝒜).symm
  exact LinearEquiv.ofLinear (gradedCommAux _ _ _) (gradedCommAux _ _ _)
    (gradedCommAux_comp_gradedCommAux _ _ _) (gradedCommAux_comp_gradedCommAux _ _ _)


/-- The braiding is symmetric. -/
@[simp]
theorem gradedComm_symm : (gradedComm R 𝒜 ℬ).symm = gradedComm R ℬ 𝒜 := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    ⊢ Eq (TensorProduct.gradedComm R 𝒜 ℬ).symm (TensorProduct.gradedComm R ℬ 𝒜)
  -/
  rw [gradedComm, gradedComm, LinearEquiv.trans_symm, LinearEquiv.symm_symm]
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    ⊢ Eq ((TensorProduct.directSum R R ℬ 𝒜).trans ((TensorProduct.directSum R R 𝒜  …
  -/
  ext
  /-
    case h
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    x✝ : TensorProduct R (DirectSum ι fun i => ℬ i) (DirectSum ι fun i => 𝒜 i)
    ⊢ Eq (((TensorProduct.directSum R R ℬ 𝒜).trans ((TensorProduct.directSum R R 𝒜 …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem gradedComm_of_tmul_of (i j : ι) (a : 𝒜 i) (b : ℬ j) :
    gradedComm R 𝒜 ℬ (lof R _ 𝒜 i a ⊗ₜ lof R _ ℬ j b) =
      (-1 : ℤˣ)^(j * i) • (lof R _ ℬ _ b ⊗ₜ lof R _ 𝒜 _ a) := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i j : ι
    a : 𝒜 i
    b : ℬ j
    ⊢ Eq ((TensorProduct.gradedComm R 𝒜 ℬ) (TensorProduct.tmul R ((DirectSum.lof R …
  -/
  rw [gradedComm]
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    i j : ι
    a : 𝒜 i
    b : ℬ j
    ⊢ Eq ((((TensorProduct.directSum R R 𝒜 ℬ).trans (LinearEquiv.ofLinear (TensorP …
  -/
  dsimp only [LinearEquiv.trans_apply, LinearEquiv.ofLinear_apply]
  rw [TensorProduct.directSum_lof_tmul_lof, gradedCommAux_lof_tmul, Units.smul_def,
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 specialized `map_smul` to `LinearEquiv.map_smul` to avoid timeouts.
    ← Int.cast_smul_eq_zsmul R, LinearEquiv.map_smul, TensorProduct.directSum_symm_lof_tmul,
    Int.cast_smul_eq_zsmul, ← Units.smul_def]


theorem gradedComm_tmul_of_zero (a : ⨁ i, 𝒜 i) (b : ℬ 0) :
    gradedComm R 𝒜 ℬ (a ⊗ₜ lof R _ ℬ 0 b) = lof R _ ℬ _ b ⊗ₜ a := by
  suffices
    (gradedComm R 𝒜 ℬ).toLinearMap ∘ₗ
        (TensorProduct.mk R (⨁ i, 𝒜 i) (⨁ i, ℬ i)).flip (lof R _ ℬ 0 b) =
      TensorProduct.mk R _ _ (lof R _ ℬ 0 b) from
    DFunLike.congr_fun this a
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    a : DirectSum ι fun i => 𝒜 i
    b : ℬ 0
    ⊢ Eq ((↑(TensorProduct.gradedComm R 𝒜 ℬ)).comp ((TensorProduct.mk R (DirectSum …
  -/
  ext i a
  /-
    case H.h
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    a✝ : DirectSum ι fun i => 𝒜 i
    b : ℬ 0
    i : ι
    a : 𝒜 i
    ⊢ Eq ((((↑(TensorProduct.gradedComm R 𝒜 ℬ)).comp ((TensorProduct.mk R (DirectS …
  -/
  dsimp
  /-
    case H.h
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    a✝ : DirectSum ι fun i => 𝒜 i
    b : ℬ 0
    i : ι
    a : 𝒜 i
    ⊢ Eq ((TensorProduct.gradedComm R 𝒜 ℬ) (TensorProduct.tmul R ((DirectSum.lof R …
  -/
  rw [gradedComm_of_tmul_of, zero_mul, uzpow_zero, one_smul]
  /-
    🎉 no goals
  -/


theorem gradedComm_of_zero_tmul (a : 𝒜 0) (b : ⨁ i, ℬ i) :
    gradedComm R 𝒜 ℬ (lof R _ 𝒜 0 a ⊗ₜ b) = b ⊗ₜ lof R _ 𝒜 _ a := by
  suffices
    (gradedComm R 𝒜 ℬ).toLinearMap ∘ₗ (TensorProduct.mk R (⨁ i, 𝒜 i) (⨁ i, ℬ i)) (lof R _ 𝒜 0 a) =
      (TensorProduct.mk R _ _).flip (lof R _ 𝒜 0 a) from
    DFunLike.congr_fun this b
  /-
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    a : 𝒜 0
    b : DirectSum ι fun i => ℬ i
    ⊢ Eq ((↑(TensorProduct.gradedComm R 𝒜 ℬ)).comp ((TensorProduct.mk R (DirectSum …
  -/
  ext i b
  /-
    case H.h
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    a : 𝒜 0
    b✝ : DirectSum ι fun i => ℬ i
    i : ι
    b : ℬ i
    ⊢ Eq ((((↑(TensorProduct.gradedComm R 𝒜 ℬ)).comp ((TensorProduct.mk R (DirectS …
  -/
  dsimp
  /-
    case H.h
    R : Type u_1
    ι : Type u_2
    inst✝⁷ : CommSemiring ι
    inst✝⁶ : Module ι (Additive (Units Int))
    inst✝⁵ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝² : (i : ι) → AddCommGroup (ℬ i)
    inst✝¹ : (i : ι) → Module R (𝒜 i)
    inst✝ : (i : ι) → Module R (ℬ i)
    a : 𝒜 0
    b✝ : DirectSum ι fun i => ℬ i
    i : ι
    b : ℬ i
    ⊢ Eq ((TensorProduct.gradedComm R 𝒜 ℬ) (TensorProduct.tmul R ((DirectSum.lof R …
  -/
  rw [gradedComm_of_tmul_of, mul_zero, uzpow_zero, one_smul]
  /-
    🎉 no goals
  -/


theorem gradedComm_tmul_one [DirectSum.GRing ℬ] (a : ⨁ i, 𝒜 i) :
    gradedComm R 𝒜 ℬ (a ⊗ₜ 1) = 1 ⊗ₜ a :=
  gradedComm_tmul_of_zero _ _ _ _ _


theorem gradedComm_one_tmul [DirectSum.GRing 𝒜] (b : ⨁ i, ℬ i) :
    gradedComm R 𝒜 ℬ (1 ⊗ₜ b) = b ⊗ₜ 1 :=
  gradedComm_of_zero_tmul _ _ _ _ _


@[simp, nolint simpNF] -- linter times out
theorem gradedComm_one [DirectSum.GRing 𝒜] [DirectSum.GRing ℬ] : gradedComm R 𝒜 ℬ 1 = 1 :=
  gradedComm_one_tmul _ _ _ _


theorem gradedComm_tmul_algebraMap [DirectSum.GRing ℬ] [DirectSum.GAlgebra R ℬ]
    (a : ⨁ i, 𝒜 i) (r : R) :
    gradedComm R 𝒜 ℬ (a ⊗ₜ algebraMap R _ r) = algebraMap R _ r ⊗ₜ a :=
  gradedComm_tmul_of_zero _ _ _ _ _


theorem gradedComm_algebraMap_tmul [DirectSum.GRing 𝒜] [DirectSum.GAlgebra R 𝒜]
    (r : R) (b : ⨁ i, ℬ i) :
    gradedComm R 𝒜 ℬ (algebraMap R _ r ⊗ₜ b) = b ⊗ₜ algebraMap R _ r :=
  gradedComm_of_zero_tmul _ _ _ _ _


theorem gradedComm_algebraMap [DirectSum.GRing 𝒜] [DirectSum.GRing ℬ]
    [DirectSum.GAlgebra R 𝒜] [DirectSum.GAlgebra R ℬ] (r : R) :
    gradedComm R 𝒜 ℬ (algebraMap R _ r) = algebraMap R _ r :=
  (gradedComm_algebraMap_tmul R 𝒜 ℬ r 1).trans (Algebra.TensorProduct.algebraMap_apply' r).symm


open TensorProduct (assoc map) in
/-- The multiplication operation for tensor products of externally `ι`-graded algebras. -/
noncomputable irreducible_def gradedMul :
    letI AB := DirectSum _ 𝒜 ⊗[R] DirectSum _ ℬ
    letI : Module R AB := TensorProduct.leftModule
    AB →ₗ[R] AB →ₗ[R] AB := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)) ( …
  -/
  refine TensorProduct.curry ?_
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R (DirectSum ι 𝒜) ( …
  -/
  refine map (LinearMap.mul' R (⨁ i, 𝒜 i)) (LinearMap.mul' R (⨁ i, ℬ i)) ∘ₗ ?_
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (TensorProduct R (DirectSum ι 𝒜) ( …
  -/
  refine (assoc R _ _ _).symm.toLinearMap ∘ₗ .lTensor _ ?_ ∘ₗ (assoc R _ _ _).toLinearMap
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι ℬ) (TensorProduct R ( …
  -/
  refine (assoc R _ _ _).toLinearMap ∘ₗ .rTensor _ ?_ ∘ₗ (assoc R _ _ _).symm.toLinearMap
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    ⊢ LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι ℬ) (DirectSum ι 𝒜)) ( …
  -/
  exact (gradedComm _ _ _).toLinearMap
  /-
    🎉 no goals
  -/


theorem tmul_of_gradedMul_of_tmul (j₁ i₂ : ι)
    (a₁ : ⨁ i, 𝒜 i) (b₁ : ℬ j₁) (a₂ : 𝒜 i₂) (b₂ : ⨁ i, ℬ i) :
    gradedMul R 𝒜 ℬ (a₁ ⊗ₜ lof R _ ℬ j₁ b₁) (lof R _ 𝒜 i₂ a₂ ⊗ₜ b₂) =
      (-1 : ℤˣ)^(j₁ * i₂) • ((a₁ * lof R _ 𝒜 _ a₂) ⊗ₜ (lof R _ ℬ _ b₁ * b₂)) := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    j₁ i₂ : ι
    a₁ : DirectSum ι fun i => 𝒜 i
    b₁ : ℬ j₁
    a₂ : 𝒜 i₂
    b₂ : DirectSum ι fun i => ℬ i
    ⊢ Eq (((TensorProduct.gradedMul R 𝒜 ℬ) (TensorProduct.tmul R a₁ ((DirectSum.lo …
  -/
  rw [gradedMul]
  dsimp only [curry_apply, LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply, assoc_tmul,
    map_tmul, LinearMap.id_coe, id_eq, assoc_symm_tmul, LinearMap.rTensor_tmul,
    LinearMap.lTensor_tmul]
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    j₁ i₂ : ι
    a₁ : DirectSum ι fun i => 𝒜 i
    b₁ : ℬ j₁
    a₂ : 𝒜 i₂
    b₂ : DirectSum ι fun i => ℬ i
    ⊢ Eq ((TensorProduct.map (LinearMap.mul' R (DirectSum ι fun i => 𝒜 i)) (Linear …
  -/
  rw [mul_comm j₁ i₂, gradedComm_of_tmul_of]
  -- the tower smul lemmas elaborate too slowly
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    j₁ i₂ : ι
    a₁ : DirectSum ι fun i => 𝒜 i
    b₁ : ℬ j₁
    a₂ : 𝒜 i₂
    b₂ : DirectSum ι fun i => ℬ i
    ⊢ Eq ((TensorProduct.map (LinearMap.mul' R (DirectSum ι fun i => 𝒜 i)) (Linear …
  -/
  rw [Units.smul_def, Units.smul_def, ← Int.cast_smul_eq_zsmul R, ← Int.cast_smul_eq_zsmul R]
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specialize `map_smul` to avoid timeouts.
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    j₁ i₂ : ι
    a₁ : DirectSum ι fun i => 𝒜 i
    b₁ : ℬ j₁
    a₂ : 𝒜 i₂
    b₂ : DirectSum ι fun i => ℬ i
    ⊢ Eq ((TensorProduct.map (LinearMap.mul' R (DirectSum ι fun i => 𝒜 i)) (Linear …
  -/
  rw [← smul_tmul', LinearEquiv.map_smul, tmul_smul, LinearEquiv.map_smul, LinearMap.map_smul]
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    j₁ i₂ : ι
    a₁ : DirectSum ι fun i => 𝒜 i
    b₁ : ℬ j₁
    a₂ : 𝒜 i₂
    b₂ : DirectSum ι fun i => ℬ i
    ⊢ Eq (HSMul.hSMul (↑↑(HPow.hPow (-1) (HMul.hMul i₂ j₁))) ((TensorProduct.map ( …
  -/
  dsimp
  /-
    🎉 no goals
  -/


theorem algebraMap_gradedMul (r : R) (x : (⨁ i, 𝒜 i) ⊗[R] (⨁ i, ℬ i)) :
    gradedMul R 𝒜 ℬ (algebraMap R _ r ⊗ₜ 1) x = r • x := by
  suffices gradedMul R 𝒜 ℬ (algebraMap R _ r ⊗ₜ 1) = DistribMulAction.toLinearMap R _ r by
    exact DFunLike.congr_fun this x
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    r : R
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ⊢ Eq ((TensorProduct.gradedMul R 𝒜 ℬ) (TensorProduct.tmul R ((algebraMap R (Di …
  -/
  ext ia a ib b
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    r : R
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ia : ι
    a : 𝒜 ia
    ib : ι
    b : ℬ ib
    ⊢ Eq (((((TensorProduct.AlgebraTensorModule.curry ((TensorProduct.gradedMul R  …
  -/
  dsimp
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    r : R
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ia : ι
    a : 𝒜 ia
    ib : ι
    b : ℬ ib
    ⊢ Eq (((TensorProduct.gradedMul R 𝒜 ℬ) (TensorProduct.tmul R ((algebraMap R (D …
  -/
  erw [tmul_of_gradedMul_of_tmul]
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    r : R
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ia : ι
    a : 𝒜 ia
    ib : ι
    b : ℬ ib
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul 0 ia)) (TensorProduct.tmul R (HMu …
  -/
  rw [zero_mul, uzpow_zero, one_smul, smul_tmul']
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    r : R
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ia : ι
    a : 𝒜 ia
    ib : ι
    b : ℬ ib
    ⊢ Eq (TensorProduct.tmul R (HMul.hMul ((algebraMap R (DirectSum ι 𝒜)) r) ((Dir …
  -/
  erw [one_mul, _root_.Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem one_gradedMul (x : (⨁ i, 𝒜 i) ⊗[R] (⨁ i, ℬ i)) :
    gradedMul R 𝒜 ℬ 1 x = x := by
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specialize `map_one` to avoid timeouts.
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ⊢ Eq (((TensorProduct.gradedMul R 𝒜 ℬ) 1) x) x
  -/
  simpa only [RingHom.map_one, one_smul] using algebraMap_gradedMul 𝒜 ℬ 1 x
  /-
    🎉 no goals
  -/


theorem gradedMul_algebraMap (x : (⨁ i, 𝒜 i) ⊗[R] (⨁ i, ℬ i)) (r : R) :
    gradedMul R 𝒜 ℬ x (algebraMap R _ r ⊗ₜ 1) = r • x := by
  suffices (gradedMul R 𝒜 ℬ).flip (algebraMap R _ r ⊗ₜ 1) = DistribMulAction.toLinearMap R _ r by
    exact DFunLike.congr_fun this x
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    r : R
    ⊢ Eq ((TensorProduct.gradedMul R 𝒜 ℬ).flip (TensorProduct.tmul R ((algebraMap  …
  -/
  ext
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    r : R
    i✝¹ : ι
    x✝¹ : 𝒜 i✝¹
    i✝ : ι
    x✝ : ℬ i✝
    ⊢ Eq (((((TensorProduct.AlgebraTensorModule.curry ((TensorProduct.gradedMul R  …
  -/
  dsimp
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    r : R
    i✝¹ : ι
    x✝¹ : 𝒜 i✝¹
    i✝ : ι
    x✝ : ℬ i✝
    ⊢ Eq (((TensorProduct.gradedMul R 𝒜 ℬ) (TensorProduct.tmul R ((DirectSum.lof R …
  -/
  erw [tmul_of_gradedMul_of_tmul]
  rw [mul_zero, uzpow_zero, one_smul, smul_tmul',
      mul_one, _root_.Algebra.smul_def, Algebra.commutes]
  /-
    case a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    r : R
    i✝¹ : ι
    x✝¹ : 𝒜 i✝¹
    i✝ : ι
    x✝ : ℬ i✝
    ⊢ Eq (TensorProduct.tmul R (HMul.hMul ((DirectSum.lof R ι 𝒜 i✝¹) x✝¹) ((Direct …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem gradedMul_one (x : (⨁ i, 𝒜 i) ⊗[R] (⨁ i, ℬ i)) :
    gradedMul R 𝒜 ℬ x 1 = x := by
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specialize `map_one` to avoid timeouts.
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x : TensorProduct R (DirectSum ι fun i => 𝒜 i) (DirectSum ι fun i => ℬ i)
    ⊢ Eq (((TensorProduct.gradedMul R 𝒜 ℬ) x) 1) x
  -/
  simpa only [RingHom.map_one, one_smul] using gradedMul_algebraMap 𝒜 ℬ x 1
  /-
    🎉 no goals
  -/


theorem gradedMul_assoc (x y z : DirectSum _ 𝒜 ⊗[R] DirectSum _ ℬ) :
    gradedMul R 𝒜 ℬ (gradedMul R 𝒜 ℬ x y) z = gradedMul R 𝒜 ℬ x (gradedMul R 𝒜 ℬ y z) := by
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y z : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    ⊢ Eq (((TensorProduct.gradedMul R 𝒜 ℬ) (((TensorProduct.gradedMul R 𝒜 ℬ) x) y) …
  -/
  let mA := gradedMul R 𝒜 ℬ
    -- restate as an equality of morphisms so that we can use `ext`
  suffices LinearMap.llcomp R _ _ _ mA ∘ₗ mA =
      (LinearMap.llcomp R _ _ _ LinearMap.lflip <| LinearMap.llcomp R _ _ _ mA.flip ∘ₗ mA).flip by
    exact DFunLike.congr_fun (DFunLike.congr_fun (DFunLike.congr_fun this x) y) z
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y z : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    mA : LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ) …
    ⊢ Eq (((LinearMap.llcomp R (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)) ( …
  -/
  ext ixa xa ixb xb iya ya iyb yb iza za izb zb
  /-
    case a.H.h.H.h.a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y z : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    mA : LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ) …
    ixa : ι
    xa : 𝒜 ixa
    ixb : ι
    xb : ℬ ixb
    iya : ι
    ya : 𝒜 iya
    iyb : ι
    yb : ℬ iyb
    iza : ι
    za : 𝒜 iza
    izb : ι
    zb : ℬ izb
    ⊢ Eq (((((TensorProduct.AlgebraTensorModule.curry (((((TensorProduct.AlgebraTe …
  -/
  dsimp [mA]
  simp_rw [tmul_of_gradedMul_of_tmul, Units.smul_def, ← Int.cast_smul_eq_zsmul R,
    LinearMap.map_smul₂, LinearMap.map_smul, DirectSum.lof_eq_of, DirectSum.of_mul_of,
    ← DirectSum.lof_eq_of R, tmul_of_gradedMul_of_tmul, DirectSum.lof_eq_of, ← DirectSum.of_mul_of,
    ← DirectSum.lof_eq_of R, mul_assoc]
  /-
    case a.H.h.H.h.a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y z : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    mA : LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ) …
    ixa : ι
    xa : 𝒜 ixa
    ixb : ι
    xb : ℬ ixb
    iya : ι
    ya : 𝒜 iya
    iyb : ι
    yb : ℬ iyb
    iza : ι
    za : 𝒜 iza
    izb : ι
    zb : ℬ izb
    ⊢ Eq (HSMul.hSMul (↑↑(HPow.hPow (-1) (HMul.hMul ixb iya))) (HSMul.hSMul (HPow. …
  -/
  simp_rw [Int.cast_smul_eq_zsmul R, ← Units.smul_def, smul_smul, ← uzpow_add, add_mul, mul_add]
  /-
    case a.H.h.H.h.a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y z : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    mA : LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ) …
    ixa : ι
    xa : 𝒜 ixa
    ixb : ι
    xb : ℬ ixb
    iya : ι
    ya : 𝒜 iya
    iyb : ι
    yb : ℬ iyb
    iza : ι
    za : 𝒜 iza
    izb : ι
    zb : ℬ izb
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HAdd.hAdd (HMul.hMul ixb iya) (HAdd.hAdd (H …
  -/
  congr 2
  /-
    case a.H.h.H.h.a.H.h.H.h.a.H.h.H.h.e_a.e_a
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y z : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    mA : LinearMap (RingHom.id R) (TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ) …
    ixa : ι
    xa : 𝒜 ixa
    ixb : ι
    xb : ℬ ixb
    iya : ι
    ya : 𝒜 iya
    iyb : ι
    yb : ℬ iyb
    iza : ι
    za : 𝒜 iza
    izb : ι
    zb : ℬ izb
    ⊢ Eq (HAdd.hAdd (HMul.hMul ixb iya) (HAdd.hAdd (HMul.hMul ixb iza) (HMul.hMul  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem gradedComm_gradedMul (x y : DirectSum _ 𝒜 ⊗[R] DirectSum _ ℬ) :
    gradedComm R 𝒜 ℬ (gradedMul R 𝒜 ℬ x y)
      = gradedMul R ℬ 𝒜 (gradedComm R 𝒜 ℬ x) (gradedComm R 𝒜 ℬ y) := by
  suffices (gradedMul R 𝒜 ℬ).compr₂ (gradedComm R 𝒜 ℬ).toLinearMap
      = (gradedMul R ℬ 𝒜 ∘ₗ (gradedComm R 𝒜 ℬ).toLinearMap).compl₂
        (gradedComm R 𝒜 ℬ).toLinearMap from
    LinearMap.congr_fun₂ this x y
  /-
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    ⊢ Eq ((TensorProduct.gradedMul R 𝒜 ℬ).compr₂ ↑(TensorProduct.gradedComm R 𝒜 ℬ) …
  -/
  ext i₁ a₁ j₁ b₁ i₂ a₂ j₂ b₂
  /-
    case a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    i₁ : ι
    a₁ : 𝒜 i₁
    j₁ : ι
    b₁ : ℬ j₁
    i₂ : ι
    a₂ : 𝒜 i₂
    j₂ : ι
    b₂ : ℬ j₂
    ⊢ Eq (((((TensorProduct.AlgebraTensorModule.curry (((((TensorProduct.AlgebraTe …
  -/
  dsimp
  /-
    case a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    i₁ : ι
    a₁ : 𝒜 i₁
    j₁ : ι
    b₁ : ℬ j₁
    i₂ : ι
    a₂ : 𝒜 i₂
    j₂ : ι
    b₂ : ℬ j₂
    ⊢ Eq ((TensorProduct.gradedComm R 𝒜 ℬ) (((TensorProduct.gradedMul R 𝒜 ℬ) (Tens …
  -/
  rw [gradedComm_of_tmul_of, gradedComm_of_tmul_of, tmul_of_gradedMul_of_tmul]
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specialize `map_smul` to avoid timeouts.
  simp_rw [Units.smul_def, ← Int.cast_smul_eq_zsmul R, LinearEquiv.map_smul, LinearMap.map_smul,
    LinearMap.smul_apply]
  simp_rw [Int.cast_smul_eq_zsmul R, ← Units.smul_def, DirectSum.lof_eq_of, DirectSum.of_mul_of,
    ← DirectSum.lof_eq_of R, gradedComm_of_tmul_of, tmul_of_gradedMul_of_tmul, smul_smul,
    DirectSum.lof_eq_of, ← DirectSum.of_mul_of, ← DirectSum.lof_eq_of R]
  /-
    case a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    i₁ : ι
    a₁ : 𝒜 i₁
    j₁ : ι
    b₁ : ℬ j₁
    i₂ : ι
    a₂ : 𝒜 i₂
    j₂ : ι
    b₂ : ℬ j₂
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (-1) (HMul.hMul j₁ i₂)) (HPow.hPow (-1 …
  -/
  simp_rw [← uzpow_add, mul_add, add_mul, mul_comm i₁ j₂]
  /-
    case a.H.h.H.h.a.H.h.H.h
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    i₁ : ι
    a₁ : 𝒜 i₁
    j₁ : ι
    b₁ : ℬ j₁
    i₂ : ι
    a₂ : 𝒜 i₂
    j₂ : ι
    b₂ : ℬ j₂
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HAdd.hAdd (HMul.hMul j₁ i₂) (HAdd.hAdd (HAd …
  -/
  congr 1
  /-
    case a.H.h.H.h.a.H.h.H.h.e_a
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    i₁ : ι
    a₁ : 𝒜 i₁
    j₁ : ι
    b₁ : ℬ j₁
    i₂ : ι
    a₂ : 𝒜 i₂
    j₂ : ι
    b₂ : ℬ j₂
    ⊢ Eq (HPow.hPow (-1) (HAdd.hAdd (HMul.hMul j₁ i₂) (HAdd.hAdd (HAdd.hAdd (HMul. …
  -/
  abel_nf
  /-
    case a.H.h.H.h.a.H.h.H.h.e_a
    R : Type u_1
    ι : Type u_2
    inst✝¹¹ : CommSemiring ι
    inst✝¹⁰ : Module ι (Additive (Units Int))
    inst✝⁹ : DecidableEq ι
    𝒜 : ι → Type u_3
    ℬ : ι → Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : (i : ι) → AddCommGroup (𝒜 i)
    inst✝⁶ : (i : ι) → AddCommGroup (ℬ i)
    inst✝⁵ : (i : ι) → Module R (𝒜 i)
    inst✝⁴ : (i : ι) → Module R (ℬ i)
    inst✝³ : DirectSum.GRing 𝒜
    inst✝² : DirectSum.GRing ℬ
    inst✝¹ : DirectSum.GAlgebra R 𝒜
    inst✝ : DirectSum.GAlgebra R ℬ
    x y : TensorProduct R (DirectSum ι 𝒜) (DirectSum ι ℬ)
    i₁ : ι
    a₁ : 𝒜 i₁
    j₁ : ι
    b₁ : ℬ j₁
    i₂ : ι
    a₂ : 𝒜 i₂
    j₂ : ι
    b₂ : ℬ j₂
    ⊢ Eq (HPow.hPow (-1) (HAdd.hAdd (HSMul.hSMul 2 (HMul.hMul j₁ i₂)) (HAdd.hAdd ( …
  -/
  rw [two_nsmul, uzpow_add, uzpow_add, Int.units_mul_self, one_mul]
  /-
    🎉 no goals
  -/


