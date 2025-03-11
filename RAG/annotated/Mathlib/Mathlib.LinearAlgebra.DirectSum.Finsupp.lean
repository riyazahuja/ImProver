/-- The tensor product of `ι →₀ M` and `N` is linearly equivalent to `ι →₀ M ⊗[R] N` -/
noncomputable def finsuppLeft :
    (ι →₀ M) ⊗[R] N ≃ₗ[R] ι →₀ M ⊗[R] N :=
  congr (finsuppLEquivDirectSum R M ι) (.refl R N) ≪≫ₗ
    directSumLeft R (fun _ ↦ M) N ≪≫ₗ (finsuppLEquivDirectSum R _ ι).symm


lemma finsuppLeft_apply_tmul (p : ι →₀ M) (n : N) :
    finsuppLeft R M N ι (p ⊗ₜ[R] n) = p.sum fun i m ↦ Finsupp.single i (m ⊗ₜ[R] n) := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    p : Finsupp ι M
    n : N
    ⊢ Eq ((TensorProduct.finsuppLeft R M N ι) (TensorProduct.tmul R p n)) (p.sum f …
  -/
  apply p.induction_linear
    /-
      case h0
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_3
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      ι : Type u_4
      inst✝ : DecidableEq ι
      p : Finsupp ι M
      n : N
      ⊢ Eq ((TensorProduct.finsuppLeft R M N ι) (TensorProduct.tmul R 0 n)) (Finsupp …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_3
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      ι : Type u_4
      inst✝ : DecidableEq ι
      p : Finsupp ι M
      n : N
      ⊢ ∀ (f g : Finsupp ι M), Eq ((TensorProduct.finsuppLeft R M N ι) (TensorProduc …
    -/
  · intros f g hf hg; simp [add_tmul, map_add, hf, hg, Finsupp.sum_add_index]
                      /-
                        🎉 no goals
                      -/
    /-
      case hsingle
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_3
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      ι : Type u_4
      inst✝ : DecidableEq ι
      p : Finsupp ι M
      n : N
      ⊢ ∀ (a : ι) (b : M), Eq ((TensorProduct.finsuppLeft R M N ι) (TensorProduct.tm …
    -/
  · simp [finsuppLeft]
    /-
      🎉 no goals
    -/


@[simp]
lemma finsuppLeft_apply_tmul_apply (p : ι →₀ M) (n : N) (i : ι) :
    finsuppLeft R M N ι (p ⊗ₜ[R] n) i = p i ⊗ₜ[R] n := by
  rw [finsuppLeft_apply_tmul, Finsupp.sum_apply,
    Finsupp.sum_eq_single i (fun _ _ ↦ Finsupp.single_eq_of_ne) (by simp), Finsupp.single_eq_same]


theorem finsuppLeft_apply (t : (ι →₀ M) ⊗[R] N) (i : ι) :
    finsuppLeft R M N ι t i = rTensor N (Finsupp.lapply i) t := by
  induction t with
  | zero => simp
  | tmul f n => simp only [finsuppLeft_apply_tmul_apply, rTensor_tmul, Finsupp.lapply_apply]
  | add x y hx hy => simp [map_add, hx, hy]


@[simp]
lemma finsuppLeft_symm_apply_single (i : ι) (m : M) (n : N) :
    (finsuppLeft R M N ι).symm (Finsupp.single i (m ⊗ₜ[R] n)) =
      Finsupp.single i m ⊗ₜ[R] n := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    i : ι
    m : M
    n : N
    ⊢ Eq ((TensorProduct.finsuppLeft R M N ι).symm (Finsupp.single i (TensorProduc …
  -/
  simp [finsuppLeft, Finsupp.lsum]
  /-
    🎉 no goals
  -/


/-- The tensor product of `M` and `ι →₀ N` is linearly equivalent to `ι →₀ M ⊗[R] N` -/
noncomputable def finsuppRight :
    M ⊗[R] (ι →₀ N) ≃ₗ[R] ι →₀ M ⊗[R] N :=
  congr (.refl R M) (finsuppLEquivDirectSum R N ι) ≪≫ₗ
    directSumRight R M (fun _ : ι ↦ N) ≪≫ₗ (finsuppLEquivDirectSum R _ ι).symm


lemma finsuppRight_apply_tmul (m : M) (p : ι →₀ N) :
    finsuppRight R M N ι (m ⊗ₜ[R] p) = p.sum fun i n ↦ Finsupp.single i (m ⊗ₜ[R] n) := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    m : M
    p : Finsupp ι N
    ⊢ Eq ((TensorProduct.finsuppRight R M N ι) (TensorProduct.tmul R m p)) (p.sum  …
  -/
  apply p.induction_linear
    /-
      case h0
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_3
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      ι : Type u_4
      inst✝ : DecidableEq ι
      m : M
      p : Finsupp ι N
      ⊢ Eq ((TensorProduct.finsuppRight R M N ι) (TensorProduct.tmul R m 0)) (Finsup …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_3
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      ι : Type u_4
      inst✝ : DecidableEq ι
      m : M
      p : Finsupp ι N
      ⊢ ∀ (f g : Finsupp ι N), Eq ((TensorProduct.finsuppRight R M N ι) (TensorProdu …
    -/
  · intros f g hf hg; simp [tmul_add, map_add, hf, hg, Finsupp.sum_add_index]
                      /-
                        🎉 no goals
                      -/
    /-
      case hsingle
      R : Type u_1
      inst✝⁵ : CommSemiring R
      M : Type u_2
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module R M
      N : Type u_3
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      ι : Type u_4
      inst✝ : DecidableEq ι
      m : M
      p : Finsupp ι N
      ⊢ ∀ (a : ι) (b : N), Eq ((TensorProduct.finsuppRight R M N ι) (TensorProduct.t …
    -/
  · simp [finsuppRight]
    /-
      🎉 no goals
    -/


@[simp]
lemma finsuppRight_apply_tmul_apply (m : M) (p : ι →₀ N) (i : ι) :
    finsuppRight R M N ι (m ⊗ₜ[R] p) i = m ⊗ₜ[R] p i := by
  rw [finsuppRight_apply_tmul, Finsupp.sum_apply,
    Finsupp.sum_eq_single i (fun _ _ ↦ Finsupp.single_eq_of_ne) (by simp), Finsupp.single_eq_same]


theorem finsuppRight_apply (t : M ⊗[R] (ι →₀ N)) (i : ι) :
    finsuppRight R M N ι t i = lTensor M (Finsupp.lapply i) t := by
  induction t with
  | zero => simp
  | tmul m f => simp [finsuppRight_apply_tmul_apply]
  | add x y hx hy => simp [map_add, hx, hy]


@[simp]
lemma finsuppRight_symm_apply_single (i : ι) (m : M) (n : N) :
    (finsuppRight R M N ι).symm (Finsupp.single i (m ⊗ₜ[R] n)) =
      m ⊗ₜ[R] Finsupp.single i n := by
  /-
    R : Type u_1
    inst✝⁵ : CommSemiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    i : ι
    m : M
    n : N
    ⊢ Eq ((TensorProduct.finsuppRight R M N ι).symm (Finsupp.single i (TensorProdu …
  -/
  simp [finsuppRight, Finsupp.lsum]
  /-
    🎉 no goals
  -/


lemma finsuppLeft_smul' (s : S) (t : (ι →₀ M) ⊗[R] N) :
    finsuppLeft R M N ι (s • t) = s • finsuppLeft R M N ι t := by
  induction t with
  | zero => simp
  | add x y hx hy => simp [hx, hy]
  | tmul p n => ext; simp [smul_tmul', finsuppLeft_apply_tmul_apply]


/-- When `M` is also an `S`-module, then `TensorProduct.finsuppLeft R M N``
  is an `S`-linear equiv -/
noncomputable def finsuppLeft' :
    (ι →₀ M) ⊗[R] N ≃ₗ[S] ι →₀ M ⊗[R] N where
  __ := finsuppLeft R M N ι
  map_smul' := finsuppLeft_smul'


lemma finsuppLeft'_apply (x : (ι →₀ M) ⊗[R] N) :
    finsuppLeft' R M N ι S x = finsuppLeft R M N ι x := rfl

/- -- TODO : reprove using the existing heterobasic lemmas
noncomputable example :
    (ι →₀ M) ⊗[R] N ≃ₗ[S] ι →₀ (M ⊗[R] N) := by
  have f : (⨁ (i₁ : ι), M) ⊗[R] N ≃ₗ[S] ⨁ (i : ι), M ⊗[R] N := sorry
  exact (AlgebraTensorModule.congr
    (finsuppLEquivDirectSum S M ι) (.refl R N)).trans
    (f.trans (finsuppLEquivDirectSum S (M ⊗[R] N) ι).symm) -/


/-- The tensor product of `ι →₀ R` and `N` is linearly equivalent to `ι →₀ N` -/
noncomputable def finsuppScalarLeft :
    (ι →₀ R) ⊗[R] N ≃ₗ[R] ι →₀ N :=
  finsuppLeft R R N ι ≪≫ₗ (Finsupp.mapRange.linearEquiv (TensorProduct.lid R N))


@[simp]
lemma finsuppScalarLeft_apply_tmul_apply (p : ι →₀ R) (n : N) (i : ι) :
    finsuppScalarLeft R N ι (p ⊗ₜ[R] n) i = p i • n := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    p : Finsupp ι R
    n : N
    i : ι
    ⊢ Eq (((TensorProduct.finsuppScalarLeft R N ι) (TensorProduct.tmul R p n)) i)  …
  -/
  simp [finsuppScalarLeft]
  /-
    🎉 no goals
  -/


lemma finsuppScalarLeft_apply_tmul (p : ι →₀ R) (n : N) :
    finsuppScalarLeft R N ι (p ⊗ₜ[R] n) = p.sum fun i m ↦ Finsupp.single i (m • n) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    p : Finsupp ι R
    n : N
    ⊢ Eq ((TensorProduct.finsuppScalarLeft R N ι) (TensorProduct.tmul R p n)) (p.s …
  -/
  ext i
  rw [finsuppScalarLeft_apply_tmul_apply, Finsupp.sum_apply,
    Finsupp.sum_eq_single i (fun _ _ ↦ Finsupp.single_eq_of_ne) (by simp), Finsupp.single_eq_same]


lemma finsuppScalarLeft_apply (pn : (ι →₀ R) ⊗[R] N) (i : ι) :
    finsuppScalarLeft R N ι pn i = TensorProduct.lid R N ((Finsupp.lapply i).rTensor N pn) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    pn : TensorProduct R (Finsupp ι R) N
    i : ι
    ⊢ Eq (((TensorProduct.finsuppScalarLeft R N ι) pn) i) ((TensorProduct.lid R N) …
  -/
  simp [finsuppScalarLeft, finsuppLeft_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma finsuppScalarLeft_symm_apply_single (i : ι) (n : N) :
    (finsuppScalarLeft R N ι).symm (Finsupp.single i n) =
      (Finsupp.single i 1) ⊗ₜ[R] n := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_4
    inst✝ : DecidableEq ι
    i : ι
    n : N
    ⊢ Eq ((TensorProduct.finsuppScalarLeft R N ι).symm (Finsupp.single i n)) (Tens …
  -/
  simp [finsuppScalarLeft, finsuppLeft_symm_apply_single]
  /-
    🎉 no goals
  -/


/-- The tensor product of `M` and `ι →₀ R` is linearly equivalent to `ι →₀ M` -/
noncomputable def finsuppScalarRight :
    M ⊗[R] (ι →₀ R) ≃ₗ[R] ι →₀ M :=
  finsuppRight R M R ι ≪≫ₗ Finsupp.mapRange.linearEquiv (TensorProduct.rid R M)


@[simp]
lemma finsuppScalarRight_apply_tmul_apply (m : M) (p : ι →₀ R) (i : ι) :
    finsuppScalarRight R M ι (m ⊗ₜ[R] p) i = p i • m := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : DecidableEq ι
    m : M
    p : Finsupp ι R
    i : ι
    ⊢ Eq (((TensorProduct.finsuppScalarRight R M ι) (TensorProduct.tmul R m p)) i) …
  -/
  simp [finsuppScalarRight]
  /-
    🎉 no goals
  -/


lemma finsuppScalarRight_apply_tmul (m : M) (p : ι →₀ R) :
    finsuppScalarRight R M ι (m ⊗ₜ[R] p) = p.sum fun i n ↦ Finsupp.single i (n • m) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : DecidableEq ι
    m : M
    p : Finsupp ι R
    ⊢ Eq ((TensorProduct.finsuppScalarRight R M ι) (TensorProduct.tmul R m p)) (p. …
  -/
  ext i
  rw [finsuppScalarRight_apply_tmul_apply, Finsupp.sum_apply,
    Finsupp.sum_eq_single i (fun _ _ ↦ Finsupp.single_eq_of_ne) (by simp), Finsupp.single_eq_same]


lemma finsuppScalarRight_apply (t : M ⊗[R] (ι →₀ R)) (i : ι) :
    finsuppScalarRight R M ι t i = TensorProduct.rid R M ((Finsupp.lapply i).lTensor M t) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : DecidableEq ι
    t : TensorProduct R M (Finsupp ι R)
    i : ι
    ⊢ Eq (((TensorProduct.finsuppScalarRight R M ι) t) i) ((TensorProduct.rid R M) …
  -/
  simp [finsuppScalarRight, finsuppRight_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma finsuppScalarRight_symm_apply_single (i : ι) (m : M) :
    (finsuppScalarRight R M ι).symm (Finsupp.single i m) =
      m ⊗ₜ[R] (Finsupp.single i 1) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : DecidableEq ι
    i : ι
    m : M
    ⊢ Eq ((TensorProduct.finsuppScalarRight R M ι).symm (Finsupp.single i m)) (Ten …
  -/
  simp [finsuppScalarRight, finsuppRight_symm_apply_single]
  /-
    🎉 no goals
  -/


open scoped Classical in
/-- The tensor product of `ι →₀ M` and `κ →₀ N` is linearly equivalent to `(ι × κ) →₀ (M ⊗ N)`. -/
def finsuppTensorFinsupp : (ι →₀ M) ⊗[R] (κ →₀ N) ≃ₗ[S] ι × κ →₀ M ⊗[R] N :=
  TensorProduct.AlgebraTensorModule.congr
    (finsuppLEquivDirectSum S M ι) (finsuppLEquivDirectSum R N κ) ≪≫ₗ
    ((TensorProduct.directSum R S (fun _ : ι => M) fun _ : κ => N) ≪≫ₗ
      (finsuppLEquivDirectSum S (M ⊗[R] N) (ι × κ)).symm)


@[simp]
theorem finsuppTensorFinsupp_single (i : ι) (m : M) (k : κ) (n : N) :
    finsuppTensorFinsupp R S M N ι κ (Finsupp.single i m ⊗ₜ Finsupp.single k n) =
      Finsupp.single (i, k) (m ⊗ₜ n) := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    i : ι
    m : M
    k : κ
    n : N
    ⊢ Eq ((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (Finsupp.single …
  -/
  simp [finsuppTensorFinsupp]
  /-
    🎉 no goals
  -/


@[simp]
theorem finsuppTensorFinsupp_apply (f : ι →₀ M) (g : κ →₀ N) (i : ι) (k : κ) :
    finsuppTensorFinsupp R S M N ι κ (f ⊗ₜ g) (i, k) = f i ⊗ₜ g k := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f : Finsupp ι M
    g : Finsupp κ N
    i : ι
    k : κ
    ⊢ Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R f g)) { fst := …
  -/
  apply Finsupp.induction_linear f
    /-
      case h0
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f : Finsupp ι M
      g : Finsupp κ N
      i : ι
      k : κ
      ⊢ Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R 0 g)) { fst := …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hadd
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f : Finsupp ι M
      g : Finsupp κ N
      i : ι
      k : κ
      ⊢ ∀ (f g_1 : Finsupp ι M), Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProd …
    -/
  · intro f₁ f₂ hf₁ hf₂
    /-
      case hadd
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f : Finsupp ι M
      g : Finsupp κ N
      i : ι
      k : κ
      f₁ f₂ : Finsupp ι M
      hf₁ : Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R f₁ g)) { f …
      hf₂ : Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R f₂ g)) { f …
      ⊢ Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (HAdd.hAdd f₁  …
    -/
    simp [add_tmul, hf₁, hf₂]
    /-
      🎉 no goals
    -/
  /-
    case hsingle
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f : Finsupp ι M
    g : Finsupp κ N
    i : ι
    k : κ
    ⊢ ∀ (a : ι) (b : M), Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tm …
  -/
  intro i' m
  /-
    case hsingle
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f : Finsupp ι M
    g : Finsupp κ N
    i : ι
    k : κ
    i' : ι
    m : M
    ⊢ Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (Finsupp.singl …
  -/
  apply Finsupp.induction_linear g
    /-
      case hsingle.h0
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f : Finsupp ι M
      g : Finsupp κ N
      i : ι
      k : κ
      i' : ι
      m : M
      ⊢ Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (Finsupp.singl …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case hsingle.hadd
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f : Finsupp ι M
      g : Finsupp κ N
      i : ι
      k : κ
      i' : ι
      m : M
      ⊢ ∀ (f g : Finsupp κ N), Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduc …
    -/
  · intro g₁ g₂ hg₁ hg₂
    /-
      case hsingle.hadd
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      ι : Type u_5
      κ : Type u_6
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f : Finsupp ι M
      g : Finsupp κ N
      i : ι
      k : κ
      i' : ι
      m : M
      g₁ g₂ : Finsupp κ N
      hg₁ : Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (Finsupp.s …
      hg₂ : Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (Finsupp.s …
      ⊢ Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tmul R (Finsupp.singl …
    -/
    simp [tmul_add, hg₁, hg₂]
    /-
      🎉 no goals
    -/
  /-
    case hsingle.hsingle
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f : Finsupp ι M
    g : Finsupp κ N
    i : ι
    k : κ
    i' : ι
    m : M
    ⊢ ∀ (a : κ) (b : N), Eq (((finsuppTensorFinsupp R S M N ι κ) (TensorProduct.tm …
  -/
  intro k' n
  classical
  simp_rw [finsuppTensorFinsupp_single, Finsupp.single_apply, Prod.mk.inj_iff, ite_and]
  split_ifs <;> simp


@[simp]
theorem finsuppTensorFinsupp_symm_single (i : ι × κ) (m : M) (n : N) :
    (finsuppTensorFinsupp R S M N ι κ).symm (Finsupp.single i (m ⊗ₜ n)) =
      Finsupp.single i.1 m ⊗ₜ Finsupp.single i.2 n :=
  Prod.casesOn i fun _ _ =>
    (LinearEquiv.symm_apply_eq _).2 (finsuppTensorFinsupp_single _ _ _ _ _ _ _ _ _ _).symm


/-- A variant of `finsuppTensorFinsupp` where the first module is the ground ring. -/
def finsuppTensorFinsuppLid : (ι →₀ R) ⊗[R] (κ →₀ N) ≃ₗ[R] ι × κ →₀ N :=
  finsuppTensorFinsupp R R R N ι κ ≪≫ₗ Finsupp.lcongr (Equiv.refl _) (TensorProduct.lid R N)


@[simp]
theorem finsuppTensorFinsuppLid_apply_apply (f : ι →₀ R) (g : κ →₀ N) (a : ι) (b : κ) :
    finsuppTensorFinsuppLid R N ι κ (f ⊗ₜ[R] g) (a, b) = f a • g b := by
  /-
    R : Type u_1
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    f : Finsupp ι R
    g : Finsupp κ N
    a : ι
    b : κ
    ⊢ Eq (((finsuppTensorFinsuppLid R N ι κ) (TensorProduct.tmul R f g)) { fst :=  …
  -/
  simp [finsuppTensorFinsuppLid]
  /-
    🎉 no goals
  -/


@[simp]
theorem finsuppTensorFinsuppLid_single_tmul_single (a : ι) (b : κ) (r : R) (n : N) :
    finsuppTensorFinsuppLid R N ι κ (Finsupp.single a r ⊗ₜ[R] Finsupp.single b n) =
      Finsupp.single (a, b) (r • n) := by
  /-
    R : Type u_1
    N : Type u_4
    ι : Type u_5
    κ : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    a : ι
    b : κ
    r : R
    n : N
    ⊢ Eq ((finsuppTensorFinsuppLid R N ι κ) (TensorProduct.tmul R (Finsupp.single  …
  -/
  simp [finsuppTensorFinsuppLid]
  /-
    🎉 no goals
  -/


@[simp]
theorem finsuppTensorFinsuppLid_symm_single_smul (i : ι × κ) (r : R) (n : N) :
    (finsuppTensorFinsuppLid R N ι κ).symm (Finsupp.single i (r • n)) =
      Finsupp.single i.1 r ⊗ₜ Finsupp.single i.2 n :=
  Prod.casesOn i fun _ _ =>
    (LinearEquiv.symm_apply_eq _).2 (finsuppTensorFinsuppLid_single_tmul_single ..).symm


/-- A variant of `finsuppTensorFinsupp` where the second module is the ground ring. -/
def finsuppTensorFinsuppRid : (ι →₀ M) ⊗[R] (κ →₀ R) ≃ₗ[R] ι × κ →₀ M :=
  finsuppTensorFinsupp R R M R ι κ ≪≫ₗ Finsupp.lcongr (Equiv.refl _) (TensorProduct.rid R M)


@[simp]
theorem finsuppTensorFinsuppRid_apply_apply (f : ι →₀ M) (g : κ →₀ R) (a : ι) (b : κ) :
    finsuppTensorFinsuppRid R M ι κ (f ⊗ₜ[R] g) (a, b) = g b • f a := by
  /-
    R : Type u_1
    M : Type u_3
    ι : Type u_5
    κ : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Finsupp ι M
    g : Finsupp κ R
    a : ι
    b : κ
    ⊢ Eq (((finsuppTensorFinsuppRid R M ι κ) (TensorProduct.tmul R f g)) { fst :=  …
  -/
  simp [finsuppTensorFinsuppRid]
  /-
    🎉 no goals
  -/


@[simp]
theorem finsuppTensorFinsuppRid_single_tmul_single (a : ι) (b : κ) (m : M) (r : R) :
    finsuppTensorFinsuppRid R M ι κ (Finsupp.single a m ⊗ₜ[R] Finsupp.single b r) =
      Finsupp.single (a, b) (r • m) := by
  /-
    R : Type u_1
    M : Type u_3
    ι : Type u_5
    κ : Type u_6
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    a : ι
    b : κ
    m : M
    r : R
    ⊢ Eq ((finsuppTensorFinsuppRid R M ι κ) (TensorProduct.tmul R (Finsupp.single  …
  -/
  simp [finsuppTensorFinsuppRid]
  /-
    🎉 no goals
  -/


@[simp]
theorem finsuppTensorFinsuppRid_symm_single_smul (i : ι × κ) (m : M) (r : R) :
    (finsuppTensorFinsuppRid R M ι κ).symm (Finsupp.single i (r • m)) =
      Finsupp.single i.1 m ⊗ₜ Finsupp.single i.2 r :=
  Prod.casesOn i fun _ _ =>
    (LinearEquiv.symm_apply_eq _).2 (finsuppTensorFinsuppRid_single_tmul_single ..).symm


/-- A variant of `finsuppTensorFinsupp` where both modules are the ground ring. -/
def finsuppTensorFinsupp' : (ι →₀ R) ⊗[R] (κ →₀ R) ≃ₗ[R] ι × κ →₀ R :=
  finsuppTensorFinsuppLid R R ι κ


@[simp]
theorem finsuppTensorFinsupp'_apply_apply (f : ι →₀ R) (g : κ →₀ R) (a : ι) (b : κ) :
    finsuppTensorFinsupp' R ι κ (f ⊗ₜ[R] g) (a, b) = f a * g b :=
  finsuppTensorFinsuppLid_apply_apply R R ι κ f g a b


@[simp]
theorem finsuppTensorFinsupp'_single_tmul_single (a : ι) (b : κ) (r₁ r₂ : R) :
    finsuppTensorFinsupp' R ι κ (Finsupp.single a r₁ ⊗ₜ[R] Finsupp.single b r₂) =
      Finsupp.single (a, b) (r₁ * r₂) :=
  finsuppTensorFinsuppLid_single_tmul_single R R ι κ a b r₁ r₂


theorem finsuppTensorFinsupp'_symm_single_mul (i : ι × κ) (r₁ r₂ : R) :
    (finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i (r₁ * r₂)) =
      Finsupp.single i.1 r₁ ⊗ₜ Finsupp.single i.2 r₂ :=
  finsuppTensorFinsuppLid_symm_single_smul R R ι κ i r₁ r₂


theorem finsuppTensorFinsupp'_symm_single_eq_single_one_tmul (i : ι × κ) (r : R) :
    (finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i r) =
      Finsupp.single i.1 1 ⊗ₜ Finsupp.single i.2 r := by
  /-
    R : Type u_1
    ι : Type u_5
    κ : Type u_6
    inst✝ : CommSemiring R
    i : Prod ι κ
    r : R
    ⊢ Eq ((finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i r)) (TensorProduct. …
  -/
  nth_rw 1 [← one_mul r]
  /-
    R : Type u_1
    ι : Type u_5
    κ : Type u_6
    inst✝ : CommSemiring R
    i : Prod ι κ
    r : R
    ⊢ Eq ((finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i (HMul.hMul 1 r))) ( …
  -/
  exact finsuppTensorFinsupp'_symm_single_mul R ι κ i _ _
  /-
    🎉 no goals
  -/


theorem finsuppTensorFinsupp'_symm_single_eq_tmul_single_one (i : ι × κ) (r : R) :
    (finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i r) =
      Finsupp.single i.1 r ⊗ₜ Finsupp.single i.2 1 := by
  /-
    R : Type u_1
    ι : Type u_5
    κ : Type u_6
    inst✝ : CommSemiring R
    i : Prod ι κ
    r : R
    ⊢ Eq ((finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i r)) (TensorProduct. …
  -/
  nth_rw 1 [← mul_one r]
  /-
    R : Type u_1
    ι : Type u_5
    κ : Type u_6
    inst✝ : CommSemiring R
    i : Prod ι κ
    r : R
    ⊢ Eq ((finsuppTensorFinsupp' R ι κ).symm (Finsupp.single i (HMul.hMul r 1))) ( …
  -/
  exact finsuppTensorFinsupp'_symm_single_mul R ι κ i _ _
  /-
    🎉 no goals
  -/


theorem finsuppTensorFinsuppLid_self :
    finsuppTensorFinsuppLid R R ι κ = finsuppTensorFinsupp' R ι κ := rfl


theorem finsuppTensorFinsuppRid_self :
    finsuppTensorFinsuppRid R R ι κ = finsuppTensorFinsupp' R ι κ := by
  rw [finsuppTensorFinsupp', finsuppTensorFinsuppLid, finsuppTensorFinsuppRid,
    TensorProduct.lid_eq_rid]

