/-- Produce a matrix with `f` applied to every pair of elements from `A` and `B`. -/
def kroneckerMap (f : α → β → γ) (A : Matrix l m α) (B : Matrix n p β) : Matrix (l × n) (m × p) γ :=
  of fun (i : l × n) (j : m × p) => f (A i.1 j.1) (B i.2 j.2)

-- TODO: set as an equation lemma for `kroneckerMap`, see https://github.com/leanprover-community/mathlib4/pull/3024

@[simp]
theorem kroneckerMap_apply (f : α → β → γ) (A : Matrix l m α) (B : Matrix n p β) (i j) :
    kroneckerMap f A B i j = f (A i.1 j.1) (B i.2 j.2) :=
  rfl


theorem kroneckerMap_transpose (f : α → β → γ) (A : Matrix l m α) (B : Matrix n p β) :
    kroneckerMap f Aᵀ Bᵀ = (kroneckerMap f A B)ᵀ :=
  ext fun _ _ => rfl


theorem kroneckerMap_map_left (f : α' → β → γ) (g : α → α') (A : Matrix l m α) (B : Matrix n p β) :
    kroneckerMap f (A.map g) B = kroneckerMap (fun a b => f (g a) b) A B :=
  ext fun _ _ => rfl


theorem kroneckerMap_map_right (f : α → β' → γ) (g : β → β') (A : Matrix l m α) (B : Matrix n p β) :
    kroneckerMap f A (B.map g) = kroneckerMap (fun a b => f a (g b)) A B :=
  ext fun _ _ => rfl


theorem kroneckerMap_map (f : α → β → γ) (g : γ → γ') (A : Matrix l m α) (B : Matrix n p β) :
    (kroneckerMap f A B).map g = kroneckerMap (fun a b => g (f a b)) A B :=
  ext fun _ _ => rfl


@[simp]
theorem kroneckerMap_zero_left [Zero α] [Zero γ] (f : α → β → γ) (hf : ∀ b, f 0 b = 0)
    (B : Matrix n p β) : kroneckerMap f (0 : Matrix l m α) B = 0 :=
  ext fun _ _ => hf _


@[simp]
theorem kroneckerMap_zero_right [Zero β] [Zero γ] (f : α → β → γ) (hf : ∀ a, f a 0 = 0)
    (A : Matrix l m α) : kroneckerMap f A (0 : Matrix n p β) = 0 :=
  ext fun _ _ => hf _


theorem kroneckerMap_add_left [Add α] [Add γ] (f : α → β → γ)
    (hf : ∀ a₁ a₂ b, f (a₁ + a₂) b = f a₁ b + f a₂ b) (A₁ A₂ : Matrix l m α) (B : Matrix n p β) :
    kroneckerMap f (A₁ + A₂) B = kroneckerMap f A₁ B + kroneckerMap f A₂ B :=
  ext fun _ _ => hf _ _ _


theorem kroneckerMap_add_right [Add β] [Add γ] (f : α → β → γ)
    (hf : ∀ a b₁ b₂, f a (b₁ + b₂) = f a b₁ + f a b₂) (A : Matrix l m α) (B₁ B₂ : Matrix n p β) :
    kroneckerMap f A (B₁ + B₂) = kroneckerMap f A B₁ + kroneckerMap f A B₂ :=
  ext fun _ _ => hf _ _ _


theorem kroneckerMap_smul_left [SMul R α] [SMul R γ] (f : α → β → γ) (r : R)
    (hf : ∀ a b, f (r • a) b = r • f a b) (A : Matrix l m α) (B : Matrix n p β) :
    kroneckerMap f (r • A) B = r • kroneckerMap f A B :=
  ext fun _ _ => hf _ _


theorem kroneckerMap_smul_right [SMul R β] [SMul R γ] (f : α → β → γ) (r : R)
    (hf : ∀ a b, f a (r • b) = r • f a b) (A : Matrix l m α) (B : Matrix n p β) :
    kroneckerMap f A (r • B) = r • kroneckerMap f A B :=
  ext fun _ _ => hf _ _


theorem kroneckerMap_diagonal_diagonal [Zero α] [Zero β] [Zero γ] [DecidableEq m] [DecidableEq n]
    (f : α → β → γ) (hf₁ : ∀ b, f 0 b = 0) (hf₂ : ∀ a, f a 0 = 0) (a : m → α) (b : n → β) :
    kroneckerMap f (diagonal a) (diagonal b) = diagonal fun mn => f (a mn.1) (b mn.2) := by
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Zero α
    inst✝³ : Zero β
    inst✝² : Zero γ
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    f : α → β → γ
    hf₁ : ∀ (b : β), Eq (f 0 b) 0
    hf₂ : ∀ (a : α), Eq (f a 0) 0
    a : m → α
    b : n → β
    ⊢ Eq (Matrix.kroneckerMap f (Matrix.diagonal a) (Matrix.diagonal b)) (Matrix.d …
  -/
  ext ⟨i₁, i₂⟩ ⟨j₁, j₂⟩
  /-
    case a.mk.mk
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Zero α
    inst✝³ : Zero β
    inst✝² : Zero γ
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    f : α → β → γ
    hf₁ : ∀ (b : β), Eq (f 0 b) 0
    hf₂ : ∀ (a : α), Eq (f a 0) 0
    a : m → α
    b : n → β
    i₁ : m
    i₂ : n
    j₁ : m
    j₂ : n
    ⊢ Eq (Matrix.kroneckerMap f (Matrix.diagonal a) (Matrix.diagonal b) { fst := i …
  -/
  simp [diagonal, apply_ite f, ite_and, ite_apply, apply_ite (f (a i₁)), hf₁, hf₂]
  /-
    🎉 no goals
  -/


theorem kroneckerMap_diagonal_right [Zero β] [Zero γ] [DecidableEq n] (f : α → β → γ)
    (hf : ∀ a, f a 0 = 0) (A : Matrix l m α) (b : n → β) :
    kroneckerMap f A (diagonal b) = blockDiagonal fun i => A.map fun a => f a (b i) := by
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    inst✝² : Zero β
    inst✝¹ : Zero γ
    inst✝ : DecidableEq n
    f : α → β → γ
    hf : ∀ (a : α), Eq (f a 0) 0
    A : Matrix l m α
    b : n → β
    ⊢ Eq (Matrix.kroneckerMap f A (Matrix.diagonal b)) (Matrix.blockDiagonal fun i …
  -/
  ext ⟨i₁, i₂⟩ ⟨j₁, j₂⟩
  /-
    case a.mk.mk
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    inst✝² : Zero β
    inst✝¹ : Zero γ
    inst✝ : DecidableEq n
    f : α → β → γ
    hf : ∀ (a : α), Eq (f a 0) 0
    A : Matrix l m α
    b : n → β
    i₁ : l
    i₂ : n
    j₁ : m
    j₂ : n
    ⊢ Eq (Matrix.kroneckerMap f A (Matrix.diagonal b) { fst := i₁, snd := i₂ } { f …
  -/
  simp [diagonal, blockDiagonal, apply_ite (f (A i₁ j₁)), hf]
  /-
    🎉 no goals
  -/


theorem kroneckerMap_diagonal_left [Zero α] [Zero γ] [DecidableEq l] (f : α → β → γ)
    (hf : ∀ b, f 0 b = 0) (a : l → α) (B : Matrix m n β) :
    kroneckerMap f (diagonal a) B =
      Matrix.reindex (Equiv.prodComm _ _) (Equiv.prodComm _ _)
        (blockDiagonal fun i => B.map fun b => f (a i) b) := by
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    inst✝² : Zero α
    inst✝¹ : Zero γ
    inst✝ : DecidableEq l
    f : α → β → γ
    hf : ∀ (b : β), Eq (f 0 b) 0
    a : l → α
    B : Matrix m n β
    ⊢ Eq (Matrix.kroneckerMap f (Matrix.diagonal a) B) ((Matrix.reindex (Equiv.pro …
  -/
  ext ⟨i₁, i₂⟩ ⟨j₁, j₂⟩
  /-
    case a.mk.mk
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    inst✝² : Zero α
    inst✝¹ : Zero γ
    inst✝ : DecidableEq l
    f : α → β → γ
    hf : ∀ (b : β), Eq (f 0 b) 0
    a : l → α
    B : Matrix m n β
    i₁ : l
    i₂ : m
    j₁ : l
    j₂ : n
    ⊢ Eq (Matrix.kroneckerMap f (Matrix.diagonal a) B { fst := i₁, snd := i₂ } { f …
  -/
  simp [diagonal, blockDiagonal, apply_ite f, ite_apply, hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem kroneckerMap_one_one [Zero α] [Zero β] [Zero γ] [One α] [One β] [One γ] [DecidableEq m]
    [DecidableEq n] (f : α → β → γ) (hf₁ : ∀ b, f 0 b = 0) (hf₂ : ∀ a, f a 0 = 0)
    (hf₃ : f 1 1 = 1) : kroneckerMap f (1 : Matrix m m α) (1 : Matrix n n β) = 1 :=
                                                             /-
                                                               α : Type u_2
                                                               β : Type u_4
                                                               γ : Type u_6
                                                               m : Type u_9
                                                               n : Type u_10
                                                               inst✝⁷ : Zero α
                                                               inst✝⁶ : Zero β
                                                               inst✝⁵ : Zero γ
                                                               inst✝⁴ : One α
                                                               inst✝³ : One β
                                                               inst✝² : One γ
                                                               inst✝¹ : DecidableEq m
                                                               inst✝ : DecidableEq n
                                                               f : α → β → γ
                                                               hf₁ : ∀ (b : β), Eq (f 0 b) 0
                                                               hf₂ : ∀ (a : α), Eq (f a 0) 0
                                                               hf₃ : Eq (f 1 1) 1
                                                               ⊢ Eq (Matrix.diagonal fun mn => f 1 1) 1
                                                             -/
  (kroneckerMap_diagonal_diagonal _ hf₁ hf₂ _ _).trans <| by simp only [hf₃, diagonal_one]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem kroneckerMap_reindex (f : α → β → γ) (el : l ≃ l') (em : m ≃ m') (en : n ≃ n') (ep : p ≃ p')
    (M : Matrix l m α) (N : Matrix n p β) :
    kroneckerMap f (reindex el em M) (reindex en ep N) =
      reindex (el.prodCongr en) (em.prodCongr ep) (kroneckerMap f M N) := by
  /-
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    p : Type u_11
    l' : Type u_14
    m' : Type u_15
    n' : Type u_16
    p' : Type u_17
    f : α → β → γ
    el : Equiv l l'
    em : Equiv m m'
    en : Equiv n n'
    ep : Equiv p p'
    M : Matrix l m α
    N : Matrix n p β
    ⊢ Eq (Matrix.kroneckerMap f ((Matrix.reindex el em) M) ((Matrix.reindex en ep) …
  -/
  ext ⟨i, i'⟩ ⟨j, j'⟩
  /-
    case a.mk.mk
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    p : Type u_11
    l' : Type u_14
    m' : Type u_15
    n' : Type u_16
    p' : Type u_17
    f : α → β → γ
    el : Equiv l l'
    em : Equiv m m'
    en : Equiv n n'
    ep : Equiv p p'
    M : Matrix l m α
    N : Matrix n p β
    i : l'
    i' : n'
    j : m'
    j' : p'
    ⊢ Eq (Matrix.kroneckerMap f ((Matrix.reindex el em) M) ((Matrix.reindex en ep) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem kroneckerMap_reindex_left (f : α → β → γ) (el : l ≃ l') (em : m ≃ m') (M : Matrix l m α)
    (N : Matrix n n' β) :
    kroneckerMap f (Matrix.reindex el em M) N =
      reindex (el.prodCongr (Equiv.refl _)) (em.prodCongr (Equiv.refl _)) (kroneckerMap f M N) :=
  kroneckerMap_reindex _ _ _ (Equiv.refl _) (Equiv.refl _) _ _


theorem kroneckerMap_reindex_right (f : α → β → γ) (em : m ≃ m') (en : n ≃ n') (M : Matrix l l' α)
    (N : Matrix m n β) :
    kroneckerMap f M (reindex em en N) =
      reindex ((Equiv.refl _).prodCongr em) ((Equiv.refl _).prodCongr en) (kroneckerMap f M N) :=
  kroneckerMap_reindex _ (Equiv.refl _) (Equiv.refl _) _ _ _ _


theorem kroneckerMap_assoc {δ ξ ω ω' : Type*} (f : α → β → γ) (g : γ → δ → ω) (f' : α → ξ → ω')
    (g' : β → δ → ξ) (A : Matrix l m α) (B : Matrix n p β) (D : Matrix q r δ) (φ : ω ≃ ω')
    (hφ : ∀ a b d, φ (g (f a b) d) = f' a (g' b d)) :
    (reindex (Equiv.prodAssoc l n q) (Equiv.prodAssoc m p r)).trans (Equiv.mapMatrix φ)
        (kroneckerMap g (kroneckerMap f A B) D) =
      kroneckerMap f' A (kroneckerMap g' B D) :=
  ext fun _ _ => hφ _ _ _


theorem kroneckerMap_assoc₁ {δ ξ ω : Type*} (f : α → β → γ) (g : γ → δ → ω) (f' : α → ξ → ω)
    (g' : β → δ → ξ) (A : Matrix l m α) (B : Matrix n p β) (D : Matrix q r δ)
    (h : ∀ a b d, g (f a b) d = f' a (g' b d)) :
    reindex (Equiv.prodAssoc l n q) (Equiv.prodAssoc m p r)
        (kroneckerMap g (kroneckerMap f A B) D) =
      kroneckerMap f' A (kroneckerMap g' B D) :=
  ext fun _ _ => h _ _ _


/-- When `f` is bilinear then `Matrix.kroneckerMap f` is also bilinear. -/
@[simps!]
def kroneckerMapBilinear [CommSemiring R] [AddCommMonoid α] [AddCommMonoid β] [AddCommMonoid γ]
    [Module R α] [Module R β] [Module R γ] (f : α →ₗ[R] β →ₗ[R] γ) :
    Matrix l m α →ₗ[R] Matrix n p β →ₗ[R] Matrix (l × n) (m × p) γ :=
  LinearMap.mk₂ R (kroneckerMap fun r s => f r s) (kroneckerMap_add_left _ <| f.map_add₂)
    (fun _ => kroneckerMap_smul_left _ _ <| f.map_smul₂ _)
    (kroneckerMap_add_right _ fun a => (f a).map_add) fun r =>
    kroneckerMap_smul_right _ _ fun a => (f a).map_smul r


/-- `Matrix.kroneckerMapBilinear` commutes with `*` if `f` does.

This is primarily used with `R = ℕ` to prove `Matrix.mul_kronecker_mul`. -/
theorem kroneckerMapBilinear_mul_mul [CommSemiring R] [Fintype m] [Fintype m']
    [NonUnitalNonAssocSemiring α] [NonUnitalNonAssocSemiring β] [NonUnitalNonAssocSemiring γ]
    [Module R α] [Module R β] [Module R γ] (f : α →ₗ[R] β →ₗ[R] γ)
    (h_comm : ∀ a b a' b', f (a * b) (a' * b') = f a a' * f b b') (A : Matrix l m α)
    (B : Matrix m n α) (A' : Matrix l' m' β) (B' : Matrix m' n' β) :
    kroneckerMapBilinear f (A * B) (A' * B') =
      kroneckerMapBilinear f A A' * kroneckerMapBilinear f B B' := by
  /-
    R : Type u_1
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    l' : Type u_14
    m' : Type u_15
    n' : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype m'
    inst✝⁵ : NonUnitalNonAssocSemiring α
    inst✝⁴ : NonUnitalNonAssocSemiring β
    inst✝³ : NonUnitalNonAssocSemiring γ
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : Module R γ
    f : LinearMap (RingHom.id R) α (LinearMap (RingHom.id R) β γ)
    h_comm : ∀ (a b : α) (a' b' : β), Eq ((f (HMul.hMul a b)) (HMul.hMul a' b')) ( …
    A : Matrix l m α
    B : Matrix m n α
    A' : Matrix l' m' β
    B' : Matrix m' n' β
    ⊢ Eq (((Matrix.kroneckerMapBilinear f) (HMul.hMul A B)) (HMul.hMul A' B')) (HM …
  -/
  ext ⟨i, i'⟩ ⟨j, j'⟩
  simp only [kroneckerMapBilinear_apply_apply, mul_apply, ← Finset.univ_product_univ,
    Finset.sum_product, kroneckerMap_apply]
  /-
    case a.mk.mk
    R : Type u_1
    α : Type u_2
    β : Type u_4
    γ : Type u_6
    l : Type u_8
    m : Type u_9
    n : Type u_10
    l' : Type u_14
    m' : Type u_15
    n' : Type u_16
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Fintype m
    inst✝⁶ : Fintype m'
    inst✝⁵ : NonUnitalNonAssocSemiring α
    inst✝⁴ : NonUnitalNonAssocSemiring β
    inst✝³ : NonUnitalNonAssocSemiring γ
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : Module R γ
    f : LinearMap (RingHom.id R) α (LinearMap (RingHom.id R) β γ)
    h_comm : ∀ (a b : α) (a' b' : β), Eq ((f (HMul.hMul a b)) (HMul.hMul a' b')) ( …
    A : Matrix l m α
    B : Matrix m n α
    A' : Matrix l' m' β
    B' : Matrix m' n' β
    i : l
    i' : l'
    j : n
    j' : n'
    ⊢ Eq ((f (Finset.univ.sum fun j_1 => HMul.hMul (A i j_1) (B j_1 j))) (Finset.u …
  -/
  simp_rw [map_sum f, LinearMap.sum_apply, map_sum, h_comm]
  /-
    🎉 no goals
  -/


/-- `trace` distributes over `Matrix.kroneckerMapBilinear`.

This is primarily used with `R = ℕ` to prove `Matrix.trace_kronecker`. -/
theorem trace_kroneckerMapBilinear [CommSemiring R] [Fintype m] [Fintype n] [AddCommMonoid α]
    [AddCommMonoid β] [AddCommMonoid γ] [Module R α] [Module R β] [Module R γ]
    (f : α →ₗ[R] β →ₗ[R] γ) (A : Matrix m m α) (B : Matrix n n β) :
    trace (kroneckerMapBilinear f A B) = f (trace A) (trace B) := by
  simp_rw [Matrix.trace, Matrix.diag, kroneckerMapBilinear_apply_apply, LinearMap.map_sum₂,
    map_sum, ← Finset.univ_product_univ, Finset.sum_product, kroneckerMap_apply]


/-- `determinant` of `Matrix.kroneckerMapBilinear`.

This is primarily used with `R = ℕ` to prove `Matrix.det_kronecker`. -/
theorem det_kroneckerMapBilinear [CommSemiring R] [Fintype m] [Fintype n] [DecidableEq m]
    [DecidableEq n] [CommRing α] [CommRing β] [CommRing γ] [Module R α] [Module R β] [Module R γ]
    (f : α →ₗ[R] β →ₗ[R] γ) (h_comm : ∀ a b a' b', f (a * b) (a' * b') = f a a' * f b b')
    (A : Matrix m m α) (B : Matrix n n β) :
    det (kroneckerMapBilinear f A B) =
      det (A.map fun a => f a 1) ^ Fintype.card n * det (B.map fun b => f 1 b) ^ Fintype.card m :=
  calc
    det (kroneckerMapBilinear f A B) =
        det (kroneckerMapBilinear f A 1 * kroneckerMapBilinear f 1 B) := by
      /-
        R : Type u_1
        α : Type u_2
        β : Type u_4
        γ : Type u_6
        m : Type u_9
        n : Type u_10
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : Fintype m
        inst✝⁸ : Fintype n
        inst✝⁷ : DecidableEq m
        inst✝⁶ : DecidableEq n
        inst✝⁵ : CommRing α
        inst✝⁴ : CommRing β
        inst✝³ : CommRing γ
        inst✝² : Module R α
        inst✝¹ : Module R β
        inst✝ : Module R γ
        f : LinearMap (RingHom.id R) α (LinearMap (RingHom.id R) β γ)
        h_comm : ∀ (a b : α) (a' b' : β), Eq ((f (HMul.hMul a b)) (HMul.hMul a' b')) ( …
        A : Matrix m m α
        B : Matrix n n β
        ⊢ Eq (((Matrix.kroneckerMapBilinear f) A) B).det (HMul.hMul (((Matrix.kronecke …
      -/
      rw [← kroneckerMapBilinear_mul_mul f h_comm, Matrix.mul_one, Matrix.one_mul]
      /-
        🎉 no goals
      -/
    _ = det (blockDiagonal fun (_ : n) => A.map fun a => f a 1) *
        det (blockDiagonal fun (_ : m) => B.map fun b => f 1 b) := by
      rw [det_mul, ← diagonal_one, ← diagonal_one, kroneckerMapBilinear_apply_apply,
        kroneckerMap_diagonal_right _ fun _ => _, kroneckerMapBilinear_apply_apply,
        kroneckerMap_diagonal_left _ fun _ => _, det_reindex_self]
        /-
          R : Type u_1
          α : Type u_2
          β : Type u_4
          γ : Type u_6
          m : Type u_9
          n : Type u_10
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : Fintype m
          inst✝⁸ : Fintype n
          inst✝⁷ : DecidableEq m
          inst✝⁶ : DecidableEq n
          inst✝⁵ : CommRing α
          inst✝⁴ : CommRing β
          inst✝³ : CommRing γ
          inst✝² : Module R α
          inst✝¹ : Module R β
          inst✝ : Module R γ
          f : LinearMap (RingHom.id R) α (LinearMap (RingHom.id R) β γ)
          h_comm : ∀ (a b : α) (a' b' : β), Eq ((f (HMul.hMul a b)) (HMul.hMul a' b')) ( …
          A : Matrix m m α
          B : Matrix n n β
          ⊢ ∀ (x : β), Eq ((f 0) x) 0
        -/
      · intro; exact LinearMap.map_zero₂ _ _
               /-
                 🎉 no goals
               -/
        /-
          R : Type u_1
          α : Type u_2
          β : Type u_4
          γ : Type u_6
          m : Type u_9
          n : Type u_10
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : Fintype m
          inst✝⁸ : Fintype n
          inst✝⁷ : DecidableEq m
          inst✝⁶ : DecidableEq n
          inst✝⁵ : CommRing α
          inst✝⁴ : CommRing β
          inst✝³ : CommRing γ
          inst✝² : Module R α
          inst✝¹ : Module R β
          inst✝ : Module R γ
          f : LinearMap (RingHom.id R) α (LinearMap (RingHom.id R) β γ)
          h_comm : ∀ (a b : α) (a' b' : β), Eq ((f (HMul.hMul a b)) (HMul.hMul a' b')) ( …
          A : Matrix m m α
          B : Matrix n n β
          ⊢ ∀ (x : α), Eq ((f x) 0) 0
        -/
      · intro; exact map_zero _
               /-
                 🎉 no goals
               -/
                /-
                  R : Type u_1
                  α : Type u_2
                  β : Type u_4
                  γ : Type u_6
                  m : Type u_9
                  n : Type u_10
                  inst✝¹⁰ : CommSemiring R
                  inst✝⁹ : Fintype m
                  inst✝⁸ : Fintype n
                  inst✝⁷ : DecidableEq m
                  inst✝⁶ : DecidableEq n
                  inst✝⁵ : CommRing α
                  inst✝⁴ : CommRing β
                  inst✝³ : CommRing γ
                  inst✝² : Module R α
                  inst✝¹ : Module R β
                  inst✝ : Module R γ
                  f : LinearMap (RingHom.id R) α (LinearMap (RingHom.id R) β γ)
                  h_comm : ∀ (a b : α) (a' b' : β), Eq ((f (HMul.hMul a b)) (HMul.hMul a' b')) ( …
                  A : Matrix m m α
                  B : Matrix n n β
                  ⊢ Eq (HMul.hMul (Matrix.blockDiagonal fun x => A.map fun a => (f a) 1).det (Ma …
                -/
    _ = _ := by simp_rw [det_blockDiagonal, Finset.prod_const, Finset.card_univ]
                /-
                  🎉 no goals
                -/


/-- The Kronecker product. This is just a shorthand for `kroneckerMap (*)`. Prefer the notation
`⊗ₖ` rather than this definition. -/
@[simp]
def kronecker [Mul α] : Matrix l m α → Matrix n p α → Matrix (l × n) (m × p) α :=
  kroneckerMap (· * ·)


scoped[Kronecker] infixl:100 " ⊗ₖ " => Matrix.kroneckerMap (· * ·)


@[simp]
theorem kronecker_apply [Mul α] (A : Matrix l m α) (B : Matrix n p α) (i₁ i₂ j₁ j₂) :
    (A ⊗ₖ B) (i₁, i₂) (j₁, j₂) = A i₁ j₁ * B i₂ j₂ :=
  rfl


/-- `Matrix.kronecker` as a bilinear map. -/
def kroneckerBilinear [CommSemiring R] [Semiring α] [Algebra R α] :
    Matrix l m α →ₗ[R] Matrix n p α →ₗ[R] Matrix (l × n) (m × p) α :=
  kroneckerMapBilinear (Algebra.lmul R α)


theorem zero_kronecker [MulZeroClass α] (B : Matrix n p α) : (0 : Matrix l m α) ⊗ₖ B = 0 :=
  kroneckerMap_zero_left _ zero_mul B


theorem kronecker_zero [MulZeroClass α] (A : Matrix l m α) : A ⊗ₖ (0 : Matrix n p α) = 0 :=
  kroneckerMap_zero_right _ mul_zero A


theorem add_kronecker [Distrib α] (A₁ A₂ : Matrix l m α) (B : Matrix n p α) :
    (A₁ + A₂) ⊗ₖ B = A₁ ⊗ₖ B + A₂ ⊗ₖ B :=
  kroneckerMap_add_left _ add_mul _ _ _


theorem kronecker_add [Distrib α] (A : Matrix l m α) (B₁ B₂ : Matrix n p α) :
    A ⊗ₖ (B₁ + B₂) = A ⊗ₖ B₁ + A ⊗ₖ B₂ :=
  kroneckerMap_add_right _ mul_add _ _ _


theorem smul_kronecker [Monoid R] [Monoid α] [MulAction R α] [IsScalarTower R α α] (r : R)
    (A : Matrix l m α) (B : Matrix n p α) : (r • A) ⊗ₖ B = r • A ⊗ₖ B :=
  kroneckerMap_smul_left _ _ (fun _ _ => smul_mul_assoc _ _ _) _ _


theorem kronecker_smul [Monoid R] [Monoid α] [MulAction R α] [SMulCommClass R α α] (r : R)
    (A : Matrix l m α) (B : Matrix n p α) : A ⊗ₖ (r • B) = r • A ⊗ₖ B :=
  kroneckerMap_smul_right _ _ (fun _ _ => mul_smul_comm _ _ _) _ _


theorem diagonal_kronecker_diagonal [MulZeroClass α] [DecidableEq m] [DecidableEq n] (a : m → α)
    (b : n → α) : diagonal a ⊗ₖ diagonal b = diagonal fun mn => a mn.1 * b mn.2 :=
  kroneckerMap_diagonal_diagonal _ zero_mul mul_zero _ _


theorem kronecker_diagonal [MulZeroClass α] [DecidableEq n] (A : Matrix l m α) (b : n → α) :
    A ⊗ₖ diagonal b = blockDiagonal fun i => A <• b i :=
  kroneckerMap_diagonal_right _ mul_zero _ _


theorem diagonal_kronecker [MulZeroClass α] [DecidableEq l] (a : l → α) (B : Matrix m n α) :
    diagonal a ⊗ₖ B =
      Matrix.reindex (Equiv.prodComm _ _) (Equiv.prodComm _ _) (blockDiagonal fun i => a i • B) :=
  kroneckerMap_diagonal_left _ zero_mul _ _


@[simp]
theorem natCast_kronecker_natCast [NonAssocSemiring α] [DecidableEq m] [DecidableEq n] (a b : ℕ) :
    (a : Matrix m m α) ⊗ₖ (b : Matrix n n α) = ↑(a * b) :=
                                                /-
                                                  α : Type u_2
                                                  m : Type u_9
                                                  n : Type u_10
                                                  inst✝² : NonAssocSemiring α
                                                  inst✝¹ : DecidableEq m
                                                  inst✝ : DecidableEq n
                                                  a b : Nat
                                                  ⊢ Eq (Matrix.diagonal fun mn => HMul.hMul ↑a ↑b) ↑(HMul.hMul a b)
                                                -/
  (diagonal_kronecker_diagonal _ _).trans <| by simp_rw [← Nat.cast_mul]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem kronecker_natCast [NonAssocSemiring α] [DecidableEq n] (A : Matrix l m α) (b : ℕ) :
    A ⊗ₖ (b : Matrix n n α) = blockDiagonal fun _ => b • A :=
  kronecker_diagonal _ _ |>.trans <| by
    /-
      α : Type u_2
      l : Type u_8
      m : Type u_9
      n : Type u_10
      inst✝¹ : NonAssocSemiring α
      inst✝ : DecidableEq n
      A : Matrix l m α
      b : Nat
      ⊢ Eq (Matrix.blockDiagonal fun i => HSMul.hSMul (MulOpposite.op ↑b) A) (Matrix …
    -/
    congr! 2
    /-
      case h.e'_7.h
      α : Type u_2
      l : Type u_8
      m : Type u_9
      n : Type u_10
      inst✝¹ : NonAssocSemiring α
      inst✝ : DecidableEq n
      A : Matrix l m α
      b : Nat
      x✝ : n
      ⊢ Eq (HSMul.hSMul (MulOpposite.op ↑b) A) (HSMul.hSMul b A)
    -/
    ext
    /-
      case h.e'_7.h.a
      α : Type u_2
      l : Type u_8
      m : Type u_9
      n : Type u_10
      inst✝¹ : NonAssocSemiring α
      inst✝ : DecidableEq n
      A : Matrix l m α
      b : Nat
      x✝ : n
      i✝ : l
      j✝ : m
      ⊢ Eq (HSMul.hSMul (MulOpposite.op ↑b) A i✝ j✝) (HSMul.hSMul b A i✝ j✝)
    -/
    simp [(Nat.cast_commute b _).eq]
    /-
      🎉 no goals
    -/


theorem natCast_kronecker [NonAssocSemiring α] [DecidableEq l] (a : ℕ) (B : Matrix m n α) :
    (a : Matrix l l α) ⊗ₖ B =
      Matrix.reindex (Equiv.prodComm _ _) (Equiv.prodComm _ _) (blockDiagonal fun _ => a • B) :=
  diagonal_kronecker _ _ |>.trans <| by
    /-
      α : Type u_2
      l : Type u_8
      m : Type u_9
      n : Type u_10
      inst✝¹ : NonAssocSemiring α
      inst✝ : DecidableEq l
      a : Nat
      B : Matrix m n α
      ⊢ Eq ((Matrix.reindex (Equiv.prodComm m l) (Equiv.prodComm n l)) (Matrix.block …
    -/
    congr! 2
    /-
      case h.e'_6.h.e'_7
      α : Type u_2
      l : Type u_8
      m : Type u_9
      n : Type u_10
      inst✝¹ : NonAssocSemiring α
      inst✝ : DecidableEq l
      a : Nat
      B : Matrix m n α
      ⊢ Eq (fun i => HSMul.hSMul (↑a) B) fun x => HSMul.hSMul a B
    -/
    ext
    /-
      case h.e'_6.h.e'_7.h.a
      α : Type u_2
      l : Type u_8
      m : Type u_9
      n : Type u_10
      inst✝¹ : NonAssocSemiring α
      inst✝ : DecidableEq l
      a : Nat
      B : Matrix m n α
      x✝ : l
      i✝ : m
      j✝ : n
      ⊢ Eq (HSMul.hSMul (↑a) B i✝ j✝) (HSMul.hSMul a B i✝ j✝)
    -/
    simp [(Nat.cast_commute a _).eq]
    /-
      🎉 no goals
    -/


theorem kronecker_ofNat [Semiring α] [DecidableEq n] (A : Matrix l m α) (b : ℕ) [b.AtLeastTwo] :
    A ⊗ₖ (no_index (OfNat.ofNat b) : Matrix n n α) =
      blockDiagonal fun _ => A <• (OfNat.ofNat b : α) :=
  kronecker_diagonal _ _


theorem ofNat_kronecker [Semiring α] [DecidableEq l] (a : ℕ) [a.AtLeastTwo] (B : Matrix m n α) :
    (no_index (OfNat.ofNat a) : Matrix l l α) ⊗ₖ B =
      Matrix.reindex (.prodComm _ _) (.prodComm _ _)
        (blockDiagonal fun _ => (OfNat.ofNat a : α) • B) :=
  diagonal_kronecker _ _


theorem one_kronecker_one [MulZeroOneClass α] [DecidableEq m] [DecidableEq n] :
    (1 : Matrix m m α) ⊗ₖ (1 : Matrix n n α) = 1 :=
  kroneckerMap_one_one _ zero_mul mul_zero (one_mul _)


theorem kronecker_one [MulZeroOneClass α] [DecidableEq n] (A : Matrix l m α) :
    A ⊗ₖ (1 : Matrix n n α) = blockDiagonal fun _ => A :=
  (kronecker_diagonal _ _).trans <| congr_arg _ <| funext fun _ => Matrix.ext fun _ _ => mul_one _


theorem one_kronecker [MulZeroOneClass α] [DecidableEq l] (B : Matrix m n α) :
    (1 : Matrix l l α) ⊗ₖ B =
      Matrix.reindex (Equiv.prodComm _ _) (Equiv.prodComm _ _) (blockDiagonal fun _ => B) :=
  (diagonal_kronecker _ _).trans <|
    congr_arg _ <| congr_arg _ <| funext fun _ => Matrix.ext fun _ _ => one_mul _


theorem mul_kronecker_mul [Fintype m] [Fintype m'] [CommSemiring α] (A : Matrix l m α)
    (B : Matrix m n α) (A' : Matrix l' m' α) (B' : Matrix m' n' α) :
    (A * B) ⊗ₖ (A' * B') = A ⊗ₖ A' * B ⊗ₖ B' :=
  kroneckerMapBilinear_mul_mul (Algebra.lmul ℕ α).toLinearMap mul_mul_mul_comm A B A' B'

-- @[simp] -- Porting note: simp-normal form is `kronecker_assoc'`

theorem kronecker_assoc [Semigroup α] (A : Matrix l m α) (B : Matrix n p α) (C : Matrix q r α) :
    reindex (Equiv.prodAssoc l n q) (Equiv.prodAssoc m p r) (A ⊗ₖ B ⊗ₖ C) = A ⊗ₖ (B ⊗ₖ C) :=
  kroneckerMap_assoc₁ _ _ _ _ A B C mul_assoc


@[simp]
theorem kronecker_assoc' [Semigroup α] (A : Matrix l m α) (B : Matrix n p α) (C : Matrix q r α) :
    submatrix (A ⊗ₖ B ⊗ₖ C) (Equiv.prodAssoc l n q).symm (Equiv.prodAssoc m p r).symm =
    A ⊗ₖ (B ⊗ₖ C) :=
  kroneckerMap_assoc₁ _ _ _ _ A B C mul_assoc


theorem trace_kronecker [Fintype m] [Fintype n] [Semiring α] (A : Matrix m m α) (B : Matrix n n α) :
    trace (A ⊗ₖ B) = trace A * trace B :=
  trace_kroneckerMapBilinear (Algebra.lmul ℕ α).toLinearMap _ _


theorem det_kronecker [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [CommRing R]
    (A : Matrix m m R) (B : Matrix n n R) :
    det (A ⊗ₖ B) = det A ^ Fintype.card n * det B ^ Fintype.card m := by
  /-
    R : Type u_1
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing R
    A : Matrix m m R
    B : Matrix n n R
    ⊢ Eq (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B).det (HMul.hMul ( …
  -/
  refine (det_kroneckerMapBilinear (Algebra.lmul ℕ R).toLinearMap mul_mul_mul_comm _ _).trans ?_
  /-
    R : Type u_1
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing R
    A : Matrix m m R
    B : Matrix n n R
    ⊢ Eq (HMul.hMul (HPow.hPow (A.map fun a => ((Algebra.lmul Nat R).toLinearMap a …
  -/
  congr 3
    /-
      case e_a.e_a.e_M
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      ⊢ Eq (A.map fun a => ((Algebra.lmul Nat R).toLinearMap a) 1) A
    -/
  · ext i j
    /-
      case e_a.e_a.e_M.a
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      i j : m
      ⊢ Eq (A.map (fun a => ((Algebra.lmul Nat R).toLinearMap a) 1) i j) (A i j)
    -/
    exact mul_one _
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_a.e_M
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      ⊢ Eq (B.map fun b => ((Algebra.lmul Nat R).toLinearMap 1) b) B
    -/
  · ext i j
    /-
      case e_a.e_a.e_M.a
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      i j : n
      ⊢ Eq (B.map (fun b => ((Algebra.lmul Nat R).toLinearMap 1) b) i j) (B i j)
    -/
    exact one_mul _
    /-
      🎉 no goals
    -/


theorem inv_kronecker [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [CommRing R]
    (A : Matrix m m R) (B : Matrix n n R) : (A ⊗ₖ B)⁻¹ = A⁻¹ ⊗ₖ B⁻¹ := by
  -- handle the special cases where either matrix is not invertible
  /-
    R : Type u_1
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing R
    A : Matrix m m R
    B : Matrix n n R
    ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
  -/
  by_cases hA : IsUnit A.det
  /-
    case pos
    R : Type u_1
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing R
    A : Matrix m m R
    B : Matrix n n R
    hA : IsUnit A.det
    ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
  -/
  swap
    /-
      case neg
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      hA : Not (IsUnit A.det)
      ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
    -/
  · cases isEmpty_or_nonempty n
      /-
        case neg.inl
        R : Type u_1
        m : Type u_9
        n : Type u_10
        inst✝⁴ : Fintype m
        inst✝³ : Fintype n
        inst✝² : DecidableEq m
        inst✝¹ : DecidableEq n
        inst✝ : CommRing R
        A : Matrix m m R
        B : Matrix n n R
        hA : Not (IsUnit A.det)
        h✝ : IsEmpty n
        ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
      -/
    · subsingleton
      /-
        🎉 no goals
      -/
    have hAB : ¬IsUnit (A ⊗ₖ B).det := by
      refine mt (fun hAB => ?_) hA
      rw [det_kronecker] at hAB
      exact (isUnit_pow_iff Fintype.card_ne_zero).mp (isUnit_of_mul_isUnit_left hAB)
    /-
      case neg.inr
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      hA : Not (IsUnit A.det)
      h✝ : Nonempty n
      hAB : Not (IsUnit (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B).det)
      ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
    -/
    rw [nonsing_inv_apply_not_isUnit _ hA, zero_kronecker, nonsing_inv_apply_not_isUnit _ hAB]
    /-
      🎉 no goals
    -/
  /-
    case pos
    R : Type u_1
    m : Type u_9
    n : Type u_10
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommRing R
    A : Matrix m m R
    B : Matrix n n R
    hA : IsUnit A.det
    ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
  -/
  by_cases hB : IsUnit B.det; swap
    /-
      case neg
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      hA : IsUnit A.det
      hB : Not (IsUnit B.det)
      ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
    -/
  · cases isEmpty_or_nonempty m
      /-
        case neg.inl
        R : Type u_1
        m : Type u_9
        n : Type u_10
        inst✝⁴ : Fintype m
        inst✝³ : Fintype n
        inst✝² : DecidableEq m
        inst✝¹ : DecidableEq n
        inst✝ : CommRing R
        A : Matrix m m R
        B : Matrix n n R
        hA : IsUnit A.det
        hB : Not (IsUnit B.det)
        h✝ : IsEmpty m
        ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
      -/
    · subsingleton
      /-
        🎉 no goals
      -/
    have hAB : ¬IsUnit (A ⊗ₖ B).det := by
      refine mt (fun hAB => ?_) hB
      rw [det_kronecker] at hAB
      exact (isUnit_pow_iff Fintype.card_ne_zero).mp (isUnit_of_mul_isUnit_right hAB)
    /-
      case neg.inr
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      hA : IsUnit A.det
      hB : Not (IsUnit B.det)
      h✝ : Nonempty m
      hAB : Not (IsUnit (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B).det)
      ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
    -/
    rw [nonsing_inv_apply_not_isUnit _ hB, kronecker_zero, nonsing_inv_apply_not_isUnit _ hAB]
    /-
      🎉 no goals
    -/
  -- otherwise follows trivially from `mul_kronecker_mul`
    /-
      case pos
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      hA : IsUnit A.det
      hB : IsUnit B.det
      ⊢ Eq (Inv.inv (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B)) (Matri …
    -/
  · apply inv_eq_right_inv
    /-
      case pos.h
      R : Type u_1
      m : Type u_9
      n : Type u_10
      inst✝⁴ : Fintype m
      inst✝³ : Fintype n
      inst✝² : DecidableEq m
      inst✝¹ : DecidableEq n
      inst✝ : CommRing R
      A : Matrix m m R
      B : Matrix n n R
      hA : IsUnit A.det
      hB : IsUnit B.det
      ⊢ Eq (HMul.hMul (Matrix.kroneckerMap (fun x1 x2 => HMul.hMul x1 x2) A B) (Matr …
    -/
    rw [← mul_kronecker_mul, ← one_kronecker_one, mul_nonsing_inv _ hA, mul_nonsing_inv _ hB]
    /-
      🎉 no goals
    -/


/-- The Kronecker tensor product. This is just a shorthand for `kroneckerMap (⊗ₜ)`.
Prefer the notation `⊗ₖₜ` rather than this definition. -/
@[simp]
def kroneckerTMul : Matrix l m α → Matrix n p β → Matrix (l × n) (m × p) (α ⊗[R] β) :=
  kroneckerMap (· ⊗ₜ ·)


scoped[Kronecker] infixl:100 " ⊗ₖₜ " => Matrix.kroneckerMap (· ⊗ₜ ·)


scoped[Kronecker]
  notation:100 x " ⊗ₖₜ[" R "] " y:100 => Matrix.kroneckerMap (TensorProduct.tmul R) x y


@[simp]
theorem kroneckerTMul_apply (A : Matrix l m α) (B : Matrix n p β) (i₁ i₂ j₁ j₂) :
    (A ⊗ₖₜ B) (i₁, i₂) (j₁, j₂) = A i₁ j₁ ⊗ₜ[R] B i₂ j₂ :=
  rfl


/-- `Matrix.kronecker` as a bilinear map. -/
def kroneckerTMulBilinear :
    Matrix l m α →ₗ[R] Matrix n p β →ₗ[R] Matrix (l × n) (m × p) (α ⊗[R] β) :=
  kroneckerMapBilinear (TensorProduct.mk R α β)


theorem zero_kroneckerTMul (B : Matrix n p β) : (0 : Matrix l m α) ⊗ₖₜ[R] B = 0 :=
  kroneckerMap_zero_left _ (zero_tmul α) B


theorem kroneckerTMul_zero (A : Matrix l m α) : A ⊗ₖₜ[R] (0 : Matrix n p β) = 0 :=
  kroneckerMap_zero_right _ (tmul_zero β) A


theorem add_kroneckerTMul (A₁ A₂ : Matrix l m α) (B : Matrix n p α) :
    (A₁ + A₂) ⊗ₖₜ[R] B = A₁ ⊗ₖₜ B + A₂ ⊗ₖₜ B :=
  kroneckerMap_add_left _ add_tmul _ _ _


theorem kroneckerTMul_add (A : Matrix l m α) (B₁ B₂ : Matrix n p α) :
    A ⊗ₖₜ[R] (B₁ + B₂) = A ⊗ₖₜ B₁ + A ⊗ₖₜ B₂ :=
  kroneckerMap_add_right _ tmul_add _ _ _


theorem smul_kroneckerTMul (r : R) (A : Matrix l m α) (B : Matrix n p α) :
    (r • A) ⊗ₖₜ[R] B = r • A ⊗ₖₜ B :=
  kroneckerMap_smul_left _ _ (fun _ _ => smul_tmul' _ _ _) _ _


theorem kroneckerTMul_smul (r : R) (A : Matrix l m α) (B : Matrix n p α) :
    A ⊗ₖₜ[R] (r • B) = r • A ⊗ₖₜ B :=
  kroneckerMap_smul_right _ _ (fun _ _ => tmul_smul _ _ _) _ _


theorem diagonal_kroneckerTMul_diagonal [DecidableEq m] [DecidableEq n] (a : m → α) (b : n → α) :
    diagonal a ⊗ₖₜ[R] diagonal b = diagonal fun mn => a mn.1 ⊗ₜ b mn.2 :=
  kroneckerMap_diagonal_diagonal _ (zero_tmul _) (tmul_zero _) _ _


theorem kroneckerTMul_diagonal [DecidableEq n] (A : Matrix l m α) (b : n → α) :
    A ⊗ₖₜ[R] diagonal b = blockDiagonal fun i => A.map fun a => a ⊗ₜ[R] b i :=
  kroneckerMap_diagonal_right _ (tmul_zero _) _ _


theorem diagonal_kroneckerTMul [DecidableEq l] (a : l → α) (B : Matrix m n α) :
    diagonal a ⊗ₖₜ[R] B =
      Matrix.reindex (Equiv.prodComm _ _) (Equiv.prodComm _ _)
        (blockDiagonal fun i => B.map fun b => a i ⊗ₜ[R] b) :=
  kroneckerMap_diagonal_left _ (zero_tmul _) _ _

-- @[simp] -- Porting note: simp-normal form is `kroneckerTMul_assoc'`

theorem kroneckerTMul_assoc (A : Matrix l m α) (B : Matrix n p β) (C : Matrix q r γ) :
    reindex (Equiv.prodAssoc l n q) (Equiv.prodAssoc m p r)
        (((A ⊗ₖₜ[R] B) ⊗ₖₜ[R] C).map (TensorProduct.assoc R α β γ)) =
      A ⊗ₖₜ[R] B ⊗ₖₜ[R] C :=
  ext fun _ _ => assoc_tmul _ _ _


@[simp]
theorem kroneckerTMul_assoc' (A : Matrix l m α) (B : Matrix n p β) (C : Matrix q r γ) :
    submatrix (((A ⊗ₖₜ[R] B) ⊗ₖₜ[R] C).map (TensorProduct.assoc R α β γ))
      (Equiv.prodAssoc l n q).symm (Equiv.prodAssoc m p r).symm = A ⊗ₖₜ[R] B ⊗ₖₜ[R] C :=
  ext fun _ _ => assoc_tmul _ _ _


theorem trace_kroneckerTMul [Fintype m] [Fintype n] (A : Matrix m m α) (B : Matrix n n β) :
    trace (A ⊗ₖₜ[R] B) = trace A ⊗ₜ[R] trace B :=
  trace_kroneckerMapBilinear (TensorProduct.mk R α β) _ _


@[simp]
theorem one_kroneckerTMul_one [DecidableEq m] [DecidableEq n] :
    (1 : Matrix m m α) ⊗ₖₜ[R] (1 : Matrix n n α) = 1 :=
  kroneckerMap_one_one _ (zero_tmul _) (tmul_zero _) rfl


unseal mul in
theorem mul_kroneckerTMul_mul [Fintype m] [Fintype m'] (A : Matrix l m α) (B : Matrix m n α)
    (A' : Matrix l' m' β) (B' : Matrix m' n' β) :
    (A * B) ⊗ₖₜ[R] (A' * B') = A ⊗ₖₜ[R] A' * B ⊗ₖₜ[R] B' :=
  kroneckerMapBilinear_mul_mul (TensorProduct.mk R α β) tmul_mul_tmul A B A' B'


unseal mul in
theorem det_kroneckerTMul [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
    (A : Matrix m m α) (B : Matrix n n β) :
    det (A ⊗ₖₜ[R] B) = (det A ^ Fintype.card n) ⊗ₜ[R] (det B ^ Fintype.card m) := by
  /-
    R : Type u_1
    α : Type u_2
    β : Type u_4
    m : Type u_9
    n : Type u_10
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing α
    inst✝⁶ : CommRing β
    inst✝⁵ : Algebra R α
    inst✝⁴ : Algebra R β
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    A : Matrix m m α
    B : Matrix n n β
    ⊢ Eq (Matrix.kroneckerMap (TensorProduct.tmul R) A B).det (TensorProduct.tmul  …
  -/
  refine (det_kroneckerMapBilinear (TensorProduct.mk R α β) tmul_mul_tmul _ _).trans ?_
  simp (config := { eta := false }) only [mk_apply, ← includeLeft_apply (S := R),
    ← includeRight_apply]
  /-
    R : Type u_1
    α : Type u_2
    β : Type u_4
    m : Type u_9
    n : Type u_10
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing α
    inst✝⁶ : CommRing β
    inst✝⁵ : Algebra R α
    inst✝⁴ : Algebra R β
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    A : Matrix m m α
    B : Matrix n n β
    ⊢ Eq (HMul.hMul (HPow.hPow (A.map fun a => Algebra.TensorProduct.includeLeft a …
  -/
  simp only [← AlgHom.mapMatrix_apply, ← AlgHom.map_det]
  simp only [includeLeft_apply, includeRight_apply, tmul_pow, tmul_mul_tmul, one_pow,
    _root_.mul_one, _root_.one_mul]


