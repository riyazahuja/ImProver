local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- The Gram-Schmidt process takes a set of vectors as input
and outputs a set of orthogonal vectors which have the same span. -/
noncomputable def gramSchmidt [WellFoundedLT ι] (f : ι → E) (n : ι) : E :=
  f n - ∑ i : Iio n, orthogonalProjection (𝕜 ∙ gramSchmidt f i) (f n)
termination_by n
/-
  ι : Type u_3
  inst✝² : LinearOrder ι
  inst✝¹ : LocallyFiniteOrderBot ι
  inst✝ : WellFoundedLT ι
  n : ι
  i : Subtype fun x => Membership.mem (Finset.Iio n) x
  ⊢ LT.lt (↑i) n
-/
decreasing_by exact mem_Iio.1 i.2
/-
  🎉 no goals
-/


/-- This lemma uses `∑ i in` instead of `∑ i :`. -/
theorem gramSchmidt_def (f : ι → E) (n : ι) :
    gramSchmidt 𝕜 f n = f n - ∑ i ∈ Iio n, orthogonalProjection (𝕜 ∙ gramSchmidt 𝕜 f i) (f n) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    ⊢ Eq (gramSchmidt 𝕜 f n) (HSub.hSub (f n) ((Finset.Iio n).sum fun i => ↑((orth …
  -/
  rw [← sum_attach, attach_eq_univ, gramSchmidt]
  /-
    🎉 no goals
  -/


theorem gramSchmidt_def' (f : ι → E) (n : ι) :
    f n = gramSchmidt 𝕜 f n + ∑ i ∈ Iio n, orthogonalProjection (𝕜 ∙ gramSchmidt 𝕜 f i) (f n) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    ⊢ Eq (f n) (HAdd.hAdd (gramSchmidt 𝕜 f n) ((Finset.Iio n).sum fun i => ↑((orth …
  -/
  rw [gramSchmidt_def, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem gramSchmidt_def'' (f : ι → E) (n : ι) :
    f n = gramSchmidt 𝕜 f n + ∑ i ∈ Iio n,
      (⟪gramSchmidt 𝕜 f i, f n⟫ / (‖gramSchmidt 𝕜 f i‖ : 𝕜) ^ 2) • gramSchmidt 𝕜 f i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    ⊢ Eq (f n) (HAdd.hAdd (gramSchmidt 𝕜 f n) ((Finset.Iio n).sum fun i => HSMul.h …
  -/
  convert gramSchmidt_def' 𝕜 f n
  /-
    case h.e'_3.h.e'_6.a
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n x✝ : ι
    a✝ : Membership.mem (Finset.Iio n) x✝
    ⊢ Eq (HSMul.hSMul (HDiv.hDiv (Inner.inner (gramSchmidt 𝕜 f x✝) (f n)) (HPow.hP …
  -/
  rw [orthogonalProjection_singleton, RCLike.ofReal_pow]
  /-
    🎉 no goals
  -/


@[simp]
theorem gramSchmidt_zero {ι : Type*} [LinearOrder ι] [LocallyFiniteOrder ι] [OrderBot ι]
    [WellFoundedLT ι] (f : ι → E) : gramSchmidt 𝕜 f ⊥ = f ⊥ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    ι : Type u_4
    inst✝³ : LinearOrder ι
    inst✝² : LocallyFiniteOrder ι
    inst✝¹ : OrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    ⊢ Eq (gramSchmidt 𝕜 f Bot.bot) (f Bot.bot)
  -/
  rw [gramSchmidt_def, Iio_eq_Ico, Finset.Ico_self, Finset.sum_empty, sub_zero]
  /-
    🎉 no goals
  -/


/-- **Gram-Schmidt Orthogonalisation**:
`gramSchmidt` produces an orthogonal system of vectors. -/
theorem gramSchmidt_orthogonal (f : ι → E) {a b : ι} (h₀ : a ≠ b) :
    ⟪gramSchmidt 𝕜 f a, gramSchmidt 𝕜 f b⟫ = 0 := by
  suffices ∀ a b : ι, a < b → ⟪gramSchmidt 𝕜 f a, gramSchmidt 𝕜 f b⟫ = 0 by
    cases' h₀.lt_or_lt with ha hb
    · exact this _ _ ha
    · rw [inner_eq_zero_symm]
      exact this _ _ hb
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    a b : ι
    h₀ : Ne a b
    ⊢ ∀ (a b : ι), LT.lt a b → Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜  …
  -/
  clear h₀ a b
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    ⊢ ∀ (a b : ι), LT.lt a b → Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜  …
  -/
  intro a b h₀
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    a b : ι
    h₀ : LT.lt a b
    ⊢ Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜 f b)) 0
  -/
  revert a
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b : ι
    ⊢ ∀ (a : ι), LT.lt a b → Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜 f  …
  -/
  apply wellFounded_lt.induction b
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b : ι
    ⊢ ∀ (x : ι), (∀ (y : ι), LT.lt y x → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (g …
  -/
  intro b ih a h₀
  simp only [gramSchmidt_def 𝕜 f b, inner_sub_right, inner_sum, orthogonalProjection_singleton,
    inner_smul_right]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b✝ b : ι
    ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
    a : ι
    h₀ : LT.lt a b
    ⊢ Eq (HSub.hSub (Inner.inner (gramSchmidt 𝕜 f a) (f b)) ((Finset.Iio b).sum fu …
  -/
  rw [Finset.sum_eq_single_of_mem a (Finset.mem_Iio.mpr h₀)]
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      b✝ b : ι
      ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
      a : ι
      h₀ : LT.lt a b
      ⊢ Eq (HSub.hSub (Inner.inner (gramSchmidt 𝕜 f a) (f b)) (HMul.hMul (HDiv.hDiv  …
    -/
  · by_cases h : gramSchmidt 𝕜 f a = 0
      /-
        case pos
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : InnerProductSpace 𝕜 E
        ι : Type u_3
        inst✝² : LinearOrder ι
        inst✝¹ : LocallyFiniteOrderBot ι
        inst✝ : WellFoundedLT ι
        f : ι → E
        b✝ b : ι
        ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
        a : ι
        h₀ : LT.lt a b
        h : Eq (gramSchmidt 𝕜 f a) 0
        ⊢ Eq (HSub.hSub (Inner.inner (gramSchmidt 𝕜 f a) (f b)) (HMul.hMul (HDiv.hDiv  …
      -/
    · simp only [h, inner_zero_left, zero_div, zero_mul, sub_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : InnerProductSpace 𝕜 E
        ι : Type u_3
        inst✝² : LinearOrder ι
        inst✝¹ : LocallyFiniteOrderBot ι
        inst✝ : WellFoundedLT ι
        f : ι → E
        b✝ b : ι
        ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
        a : ι
        h₀ : LT.lt a b
        h : Not (Eq (gramSchmidt 𝕜 f a) 0)
        ⊢ Eq (HSub.hSub (Inner.inner (gramSchmidt 𝕜 f a) (f b)) (HMul.hMul (HDiv.hDiv  …
      -/
    · rw [RCLike.ofReal_pow, ← inner_self_eq_norm_sq_to_K, div_mul_cancel₀, sub_self]
      /-
        case neg.h
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : InnerProductSpace 𝕜 E
        ι : Type u_3
        inst✝² : LinearOrder ι
        inst✝¹ : LocallyFiniteOrderBot ι
        inst✝ : WellFoundedLT ι
        f : ι → E
        b✝ b : ι
        ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
        a : ι
        h₀ : LT.lt a b
        h : Not (Eq (gramSchmidt 𝕜 f a) 0)
        ⊢ Ne (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜 f a)) 0
      -/
      rwa [inner_self_ne_zero]
      /-
        🎉 no goals
      -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b✝ b : ι
    ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
    a : ι
    h₀ : LT.lt a b
    ⊢ ∀ (b_1 : ι), Membership.mem (Finset.Iio b) b_1 → Ne b_1 a → Eq (HMul.hMul (H …
  -/
  intro i hi hia
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b✝ b : ι
    ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
    a : ι
    h₀ : LT.lt a b
    i : ι
    hi : Membership.mem (Finset.Iio b) i
    hia : Ne i a
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Inner.inner (gramSchmidt 𝕜 f i) (f b)) ↑(HPow.hPow …
  -/
  simp only [mul_eq_zero, div_eq_zero_iff, inner_self_eq_zero]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b✝ b : ι
    ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
    a : ι
    h₀ : LT.lt a b
    i : ι
    hi : Membership.mem (Finset.Iio b) i
    hia : Ne i a
    ⊢ Or (Or (Eq (Inner.inner (gramSchmidt 𝕜 f i) (f b)) 0) (Eq (↑(HPow.hPow (Norm …
  -/
  right
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    b✝ b : ι
    ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
    a : ι
    h₀ : LT.lt a b
    i : ι
    hi : Membership.mem (Finset.Iio b) i
    hia : Ne i a
    ⊢ Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜 f i)) 0
  -/
  cases' hia.lt_or_lt with hia₁ hia₂
    /-
      case h.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      b✝ b : ι
      ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
      a : ι
      h₀ : LT.lt a b
      i : ι
      hi : Membership.mem (Finset.Iio b) i
      hia : Ne i a
      hia₁ : LT.lt i a
      ⊢ Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜 f i)) 0
    -/
  · rw [inner_eq_zero_symm]
    /-
      case h.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      b✝ b : ι
      ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
      a : ι
      h₀ : LT.lt a b
      i : ι
      hi : Membership.mem (Finset.Iio b) i
      hia : Ne i a
      hia₁ : LT.lt i a
      ⊢ Eq (Inner.inner (gramSchmidt 𝕜 f i) (gramSchmidt 𝕜 f a)) 0
    -/
    exact ih a h₀ i hia₁
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      b✝ b : ι
      ih : ∀ (y : ι), LT.lt y b → ∀ (a : ι), LT.lt a y → Eq (Inner.inner (gramSchmid …
      a : ι
      h₀ : LT.lt a b
      i : ι
      hi : Membership.mem (Finset.Iio b) i
      hia : Ne i a
      hia₂ : LT.lt a i
      ⊢ Eq (Inner.inner (gramSchmidt 𝕜 f a) (gramSchmidt 𝕜 f i)) 0
    -/
  · exact ih i (mem_Iio.1 hi) a hia₂
    /-
      🎉 no goals
    -/


/-- This is another version of `gramSchmidt_orthogonal` using `Pairwise` instead. -/
theorem gramSchmidt_pairwise_orthogonal (f : ι → E) :
    Pairwise fun a b => ⟪gramSchmidt 𝕜 f a, gramSchmidt 𝕜 f b⟫ = 0 := fun _ _ =>
  gramSchmidt_orthogonal 𝕜 f


theorem gramSchmidt_inv_triangular (v : ι → E) {i j : ι} (hij : i < j) :
    ⟪gramSchmidt 𝕜 v j, v i⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    ⊢ Eq (Inner.inner (gramSchmidt 𝕜 v j) (v i)) 0
  -/
  rw [gramSchmidt_def'' 𝕜 v]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    ⊢ Eq (Inner.inner (gramSchmidt 𝕜 v j) (HAdd.hAdd (gramSchmidt 𝕜 v i) ((Finset. …
  -/
  simp only [inner_add_right, inner_sum, inner_smul_right]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    ⊢ Eq (HAdd.hAdd (Inner.inner (gramSchmidt 𝕜 v j) (gramSchmidt 𝕜 v i)) ((Finset …
  -/
  set b : ι → E := gramSchmidt 𝕜 v
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    b : ι → E := gramSchmidt 𝕜 v
    ⊢ Eq (HAdd.hAdd (Inner.inner (b j) (b i)) ((Finset.Iio i).sum fun x => HMul.hM …
  -/
  convert zero_add (0 : 𝕜)
    /-
      case h.e'_2.h.e'_5
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      v : ι → E
      i j : ι
      hij : LT.lt i j
      b : ι → E := gramSchmidt 𝕜 v
      ⊢ Eq (Inner.inner (b j) (b i)) 0
    -/
  · exact gramSchmidt_orthogonal 𝕜 v hij.ne'
    /-
      🎉 no goals
    -/
  /-
    case h.e'_2.h.e'_6
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    b : ι → E := gramSchmidt 𝕜 v
    ⊢ Eq ((Finset.Iio i).sum fun x => HMul.hMul (HDiv.hDiv (Inner.inner (b x) (v i …
  -/
  apply Finset.sum_eq_zero
  /-
    case h.e'_2.h.e'_6.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    b : ι → E := gramSchmidt 𝕜 v
    ⊢ ∀ (x : ι), Membership.mem (Finset.Iio i) x → Eq (HMul.hMul (HDiv.hDiv (Inner …
  -/
  rintro k hki'
  /-
    case h.e'_2.h.e'_6.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    b : ι → E := gramSchmidt 𝕜 v
    k : ι
    hki' : Membership.mem (Finset.Iio i) k
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Inner.inner (b k) (v i)) (HPow.hPow (↑(Norm.norm ( …
  -/
  have hki : k < i := by simpa using hki'
  /-
    case h.e'_2.h.e'_6.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    b : ι → E := gramSchmidt 𝕜 v
    k : ι
    hki' : Membership.mem (Finset.Iio i) k
    hki : LT.lt k i
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Inner.inner (b k) (v i)) (HPow.hPow (↑(Norm.norm ( …
  -/
  have : ⟪b j, b k⟫ = 0 := gramSchmidt_orthogonal 𝕜 v (hki.trans hij).ne'
  /-
    case h.e'_2.h.e'_6.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    v : ι → E
    i j : ι
    hij : LT.lt i j
    b : ι → E := gramSchmidt 𝕜 v
    k : ι
    hki' : Membership.mem (Finset.Iio i) k
    hki : LT.lt k i
    this : Eq (Inner.inner (b j) (b k)) 0
    ⊢ Eq (HMul.hMul (HDiv.hDiv (Inner.inner (b k) (v i)) (HPow.hPow (↑(Norm.norm ( …
  -/
  simp [this]
  /-
    🎉 no goals
  -/


theorem mem_span_gramSchmidt (f : ι → E) {i j : ι} (hij : i ≤ j) :
    f i ∈ span 𝕜 (gramSchmidt 𝕜 f '' Set.Iic j) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    i j : ι
    hij : LE.le i j
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image (gramSchmidt 𝕜 f) (Set.Iic j)))  …
  -/
  rw [gramSchmidt_def' 𝕜 f i]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    i j : ι
    hij : LE.le i j
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image (gramSchmidt 𝕜 f) (Set.Iic j)))  …
  -/
  simp_rw [orthogonalProjection_singleton]
  exact Submodule.add_mem _ (subset_span <| mem_image_of_mem _ hij)
    (Submodule.sum_mem _ fun k hk => smul_mem (span 𝕜 (gramSchmidt 𝕜 f '' Set.Iic j)) _ <|
      subset_span <| mem_image_of_mem (gramSchmidt 𝕜 f) <| (Finset.mem_Iio.1 hk).le.trans hij)


theorem gramSchmidt_mem_span (f : ι → E) :
    ∀ {j i}, i ≤ j → gramSchmidt 𝕜 f i ∈ span 𝕜 (f '' Set.Iic j) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    ⊢ ∀ {j i : ι}, LE.le i j → Membership.mem (Submodule.span 𝕜 (Set.image f (Set. …
  -/
  intro j i hij
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    j i : ι
    hij : LE.le i j
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f (Set.Iic j))) (gramSchmidt 𝕜 f …
  -/
  rw [gramSchmidt_def 𝕜 f i]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    j i : ι
    hij : LE.le i j
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f (Set.Iic j))) (HSub.hSub (f i) …
  -/
  simp_rw [orthogonalProjection_singleton]
  refine Submodule.sub_mem _ (subset_span (mem_image_of_mem _ hij))
    (Submodule.sum_mem _ fun k hk => ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    j i : ι
    hij : LE.le i j
    k : ι
    hk : Membership.mem (Finset.Iio i) k
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f (Set.Iic j))) (HSMul.hSMul (HD …
  -/
  let hkj : k < j := (Finset.mem_Iio.1 hk).trans_le hij
  exact smul_mem _ _
    (span_mono (image_subset f <| Set.Iic_subset_Iic.2 hkj.le) <| gramSchmidt_mem_span _ le_rfl)
termination_by j => j


theorem span_gramSchmidt_Iic (f : ι → E) (c : ι) :
    span 𝕜 (gramSchmidt 𝕜 f '' Set.Iic c) = span 𝕜 (f '' Set.Iic c) :=
  span_eq_span (Set.image_subset_iff.2 fun _ => gramSchmidt_mem_span _ _) <|
    Set.image_subset_iff.2 fun _ => mem_span_gramSchmidt _ _


theorem span_gramSchmidt_Iio (f : ι → E) (c : ι) :
    span 𝕜 (gramSchmidt 𝕜 f '' Set.Iio c) = span 𝕜 (f '' Set.Iio c) :=
  span_eq_span (Set.image_subset_iff.2 fun _ hi =>
    span_mono (image_subset _ <| Iic_subset_Iio.2 hi) <| gramSchmidt_mem_span _ _ le_rfl) <|
      Set.image_subset_iff.2 fun _ hi =>
        span_mono (image_subset _ <| Iic_subset_Iio.2 hi) <| mem_span_gramSchmidt _ _ le_rfl


/-- `gramSchmidt` preserves span of vectors. -/
theorem span_gramSchmidt (f : ι → E) : span 𝕜 (range (gramSchmidt 𝕜 f)) = span 𝕜 (range f) :=
  span_eq_span (range_subset_iff.2 fun _ =>
    span_mono (image_subset_range _ _) <| gramSchmidt_mem_span _ _ le_rfl) <|
      range_subset_iff.2 fun _ =>
        span_mono (image_subset_range _ _) <| mem_span_gramSchmidt _ _ le_rfl


theorem gramSchmidt_of_orthogonal {f : ι → E} (hf : Pairwise fun i j => ⟪f i, f j⟫ = 0) :
    gramSchmidt 𝕜 f = f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
    ⊢ Eq (gramSchmidt 𝕜 f) f
  -/
  ext i
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
    i : ι
    ⊢ Eq (gramSchmidt 𝕜 f i) (f i)
  -/
  rw [gramSchmidt_def]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
    i : ι
    ⊢ Eq (HSub.hSub (f i) ((Finset.Iio i).sum fun i_1 => ↑((orthogonalProjection ( …
  -/
  trans f i - 0
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i : ι
      ⊢ Eq (HSub.hSub (f i) ((Finset.Iio i).sum fun i_1 => ↑((orthogonalProjection ( …
    -/
  · congr
    /-
      case e_a
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i : ι
      ⊢ Eq ((Finset.Iio i).sum fun i_1 => ↑((orthogonalProjection (Submodule.span 𝕜  …
    -/
    apply Finset.sum_eq_zero
    /-
      case e_a.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i : ι
      ⊢ ∀ (x : ι), Membership.mem (Finset.Iio i) x → Eq (↑((orthogonalProjection (Su …
    -/
    intro j hj
    /-
      case e_a.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i j : ι
      hj : Membership.mem (Finset.Iio i) j
      ⊢ Eq (↑((orthogonalProjection (Submodule.span 𝕜 (Singleton.singleton (gramSchm …
    -/
    rw [Submodule.coe_eq_zero]
    suffices span 𝕜 (f '' Set.Iic j) ⟂ 𝕜 ∙ f i by
      apply orthogonalProjection_mem_subspace_orthogonalComplement_eq_zero
      rw [mem_orthogonal_singleton_iff_inner_left]
      rw [← mem_orthogonal_singleton_iff_inner_right]
      exact this (gramSchmidt_mem_span 𝕜 f (le_refl j))
    /-
      case e_a.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i j : ι
      hj : Membership.mem (Finset.Iio i) j
      ⊢ (Submodule.span 𝕜 (Set.image f (Set.Iic j))).IsOrtho (Submodule.span 𝕜 (Sing …
    -/
    rw [isOrtho_span]
    /-
      case e_a.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i j : ι
      hj : Membership.mem (Finset.Iio i) j
      ⊢ ∀ ⦃u : E⦄, Membership.mem (Set.image f (Set.Iic j)) u → ∀ ⦃v : E⦄, Membershi …
    -/
    rintro u ⟨k, hk, rfl⟩ v (rfl : v = f i)
    /-
      case e_a.h.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i j : ι
      hj : Membership.mem (Finset.Iio i) j
      k : ι
      hk : Membership.mem (Set.Iic j) k
      ⊢ Eq (Inner.inner (f k) (f i)) 0
    -/
    apply hf
    /-
      case e_a.h.intro.intro.a
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i j : ι
      hj : Membership.mem (Finset.Iio i) j
      k : ι
      hk : Membership.mem (Set.Iic j) k
      ⊢ Ne k i
    -/
    exact (lt_of_le_of_lt hk (Finset.mem_Iio.mp hj)).ne
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
      i : ι
      ⊢ Eq (HSub.hSub (f i) 0) (f i)
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem gramSchmidt_ne_zero_coe {f : ι → E} (n : ι)
    (h₀ : LinearIndependent 𝕜 (f ∘ ((↑) : Set.Iic n → ι))) : gramSchmidt 𝕜 f n ≠ 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    h₀ : LinearIndependent 𝕜 (Function.comp f Subtype.val)
    ⊢ Ne (gramSchmidt 𝕜 f n) 0
  -/
  by_contra h
  have h₁ : f n ∈ span 𝕜 (f '' Set.Iio n) := by
    rw [← span_gramSchmidt_Iio 𝕜 f n, gramSchmidt_def' 𝕜 f, h, zero_add]
    apply Submodule.sum_mem _ _
    intro a ha
    simp only [Set.mem_image, Set.mem_Iio, orthogonalProjection_singleton]
    apply Submodule.smul_mem _ _ _
    rw [Finset.mem_Iio] at ha
    exact subset_span ⟨a, ha, by rfl⟩
  have h₂ : (f ∘ ((↑) : Set.Iic n → ι)) ⟨n, le_refl n⟩ ∈
      span 𝕜 (f ∘ ((↑) : Set.Iic n → ι) '' Set.Iio ⟨n, le_refl n⟩) := by
    rw [image_comp]
    simpa using h₁
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    h₀ : LinearIndependent 𝕜 (Function.comp f Subtype.val)
    h : Eq (gramSchmidt 𝕜 f n) 0
    h₁ : Membership.mem (Submodule.span 𝕜 (Set.image f (Set.Iio n))) (f n)
    h₂ : Membership.mem (Submodule.span 𝕜 (Set.image (Function.comp f Subtype.val) …
    ⊢ False
  -/
  apply LinearIndependent.not_mem_span_image h₀ _ h₂
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    h₀ : LinearIndependent 𝕜 (Function.comp f Subtype.val)
    h : Eq (gramSchmidt 𝕜 f n) 0
    h₁ : Membership.mem (Submodule.span 𝕜 (Set.image f (Set.Iio n))) (f n)
    h₂ : Membership.mem (Submodule.span 𝕜 (Set.image (Function.comp f Subtype.val) …
    ⊢ Not (Membership.mem (Set.Iio ⟨n, ⋯⟩) ⟨n, ⋯⟩)
  -/
  simp only [Set.mem_Iio, lt_self_iff_false, not_false_iff]
  /-
    🎉 no goals
  -/


/-- If the input vectors of `gramSchmidt` are linearly independent,
then the output vectors are non-zero. -/
theorem gramSchmidt_ne_zero {f : ι → E} (n : ι) (h₀ : LinearIndependent 𝕜 f) :
    gramSchmidt 𝕜 f n ≠ 0 :=
  gramSchmidt_ne_zero_coe _ (LinearIndependent.comp h₀ _ Subtype.coe_injective)


/-- `gramSchmidt` produces a triangular matrix of vectors when given a basis. -/
theorem gramSchmidt_triangular {i j : ι} (hij : i < j) (b : Basis ι 𝕜 E) :
    b.repr (gramSchmidt 𝕜 b i) j = 0 := by
  have : gramSchmidt 𝕜 b i ∈ span 𝕜 (gramSchmidt 𝕜 b '' Set.Iio j) :=
    subset_span ((Set.mem_image _ _ _).2 ⟨i, hij, rfl⟩)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    i j : ι
    hij : LT.lt i j
    b : Basis ι 𝕜 E
    this : Membership.mem (Submodule.span 𝕜 (Set.image (gramSchmidt 𝕜 ⇑b) (Set.Iio …
    ⊢ Eq ((b.repr (gramSchmidt 𝕜 (⇑b) i)) j) 0
  -/
  have : gramSchmidt 𝕜 b i ∈ span 𝕜 (b '' Set.Iio j) := by rwa [← span_gramSchmidt_Iio 𝕜 b j]
  have : ↑(b.repr (gramSchmidt 𝕜 b i)).support ⊆ Set.Iio j :=
    Basis.repr_support_subset_of_mem_span b (Set.Iio j) this
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    i j : ι
    hij : LT.lt i j
    b : Basis ι 𝕜 E
    this✝¹ : Membership.mem (Submodule.span 𝕜 (Set.image (gramSchmidt 𝕜 ⇑b) (Set.I …
    this✝ : Membership.mem (Submodule.span 𝕜 (Set.image (⇑b) (Set.Iio j))) (gramSc …
    this : HasSubset.Subset (↑(b.repr (gramSchmidt 𝕜 (⇑b) i)).support) (Set.Iio j)
    ⊢ Eq ((b.repr (gramSchmidt 𝕜 (⇑b) i)) j) 0
  -/
  exact (Finsupp.mem_supported' _ _).1 ((Finsupp.mem_supported 𝕜 _).2 this) j Set.not_mem_Iio_self
  /-
    🎉 no goals
  -/


/-- `gramSchmidt` produces linearly independent vectors when given linearly independent vectors. -/
theorem gramSchmidt_linearIndependent {f : ι → E} (h₀ : LinearIndependent 𝕜 f) :
    LinearIndependent 𝕜 (gramSchmidt 𝕜 f) :=
  linearIndependent_of_ne_zero_of_inner_eq_zero (fun _ => gramSchmidt_ne_zero _ h₀) fun _ _ =>
    gramSchmidt_orthogonal 𝕜 f


/-- When given a basis, `gramSchmidt` produces a basis. -/
noncomputable def gramSchmidtBasis (b : Basis ι 𝕜 E) : Basis ι 𝕜 E :=
  Basis.mk (gramSchmidt_linearIndependent b.linearIndependent)
    ((span_gramSchmidt 𝕜 b).trans b.span_eq).ge


theorem coe_gramSchmidtBasis (b : Basis ι 𝕜 E) : (gramSchmidtBasis b : ι → E) = gramSchmidt 𝕜 b :=
  Basis.coe_mk _ _


/-- the normalized `gramSchmidt`
(i.e each vector in `gramSchmidtNormed` has unit length.) -/
noncomputable def gramSchmidtNormed (f : ι → E) (n : ι) : E :=
  (‖gramSchmidt 𝕜 f n‖ : 𝕜)⁻¹ • gramSchmidt 𝕜 f n


theorem gramSchmidtNormed_unit_length_coe {f : ι → E} (n : ι)
    (h₀ : LinearIndependent 𝕜 (f ∘ ((↑) : Set.Iic n → ι))) : ‖gramSchmidtNormed 𝕜 f n‖ = 1 := by
  simp only [gramSchmidt_ne_zero_coe n h₀, gramSchmidtNormed, norm_smul_inv_norm, Ne,
    not_false_iff]


theorem gramSchmidtNormed_unit_length {f : ι → E} (n : ι) (h₀ : LinearIndependent 𝕜 f) :
    ‖gramSchmidtNormed 𝕜 f n‖ = 1 :=
  gramSchmidtNormed_unit_length_coe _ (LinearIndependent.comp h₀ _ Subtype.coe_injective)


theorem gramSchmidtNormed_unit_length' {f : ι → E} {n : ι} (hn : gramSchmidtNormed 𝕜 f n ≠ 0) :
    ‖gramSchmidtNormed 𝕜 f n‖ = 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    hn : Ne (gramSchmidtNormed 𝕜 f n) 0
    ⊢ Eq (Norm.norm (gramSchmidtNormed 𝕜 f n)) 1
  -/
  rw [gramSchmidtNormed] at *
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    hn : Ne (HSMul.hSMul (Inv.inv ↑(Norm.norm (gramSchmidt 𝕜 f n))) (gramSchmidt 𝕜 …
    ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv ↑(Norm.norm (gramSchmidt 𝕜 f n))) (gramS …
  -/
  rw [norm_smul_inv_norm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    n : ι
    hn : Ne (HSMul.hSMul (Inv.inv ↑(Norm.norm (gramSchmidt 𝕜 f n))) (gramSchmidt 𝕜 …
    ⊢ Ne (gramSchmidt 𝕜 f n) 0
  -/
  simpa using hn
  /-
    🎉 no goals
  -/


/-- **Gram-Schmidt Orthonormalization**:
`gramSchmidtNormed` applied to a linearly independent set of vectors produces an orthornormal
system of vectors. -/
theorem gramSchmidt_orthonormal {f : ι → E} (h₀ : LinearIndependent 𝕜 f) :
    Orthonormal 𝕜 (gramSchmidtNormed 𝕜 f) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    h₀ : LinearIndependent 𝕜 f
    ⊢ Orthonormal 𝕜 (gramSchmidtNormed 𝕜 f)
  -/
  unfold Orthonormal
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    h₀ : LinearIndependent 𝕜 f
    ⊢ And (∀ (i : ι), Eq (Norm.norm (gramSchmidtNormed 𝕜 f i)) 1) (Pairwise fun i  …
  -/
  constructor
    /-
      case left
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      h₀ : LinearIndependent 𝕜 f
      ⊢ ∀ (i : ι), Eq (Norm.norm (gramSchmidtNormed 𝕜 f i)) 1
    -/
  · simp only [gramSchmidtNormed_unit_length, h₀, eq_self_iff_true, imp_true_iff]
    /-
      🎉 no goals
    -/
    /-
      case right
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      h₀ : LinearIndependent 𝕜 f
      ⊢ Pairwise fun i j => Eq (Inner.inner (gramSchmidtNormed 𝕜 f i) (gramSchmidtNo …
    -/
  · intro i j hij
    simp only [gramSchmidtNormed, inner_smul_left, inner_smul_right, RCLike.conj_inv,
      RCLike.conj_ofReal, mul_eq_zero, inv_eq_zero, RCLike.ofReal_eq_zero, norm_eq_zero]
    /-
      case right
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      h₀ : LinearIndependent 𝕜 f
      i j : ι
      hij : Ne i j
      ⊢ Or (Eq (gramSchmidt 𝕜 f j) 0) (Or (Eq (gramSchmidt 𝕜 f i) 0) (Eq (Inner.inne …
    -/
    repeat' right
    /-
      case right.h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      h₀ : LinearIndependent 𝕜 f
      i j : ι
      hij : Ne i j
      ⊢ Eq (Inner.inner (gramSchmidt 𝕜 f i) (gramSchmidt 𝕜 f j)) 0
    -/
    exact gramSchmidt_orthogonal 𝕜 f hij
    /-
      🎉 no goals
    -/


/-- **Gram-Schmidt Orthonormalization**:
`gramSchmidtNormed` produces an orthornormal system of vectors after removing the vectors which
become zero in the process. -/
theorem gramSchmidt_orthonormal' (f : ι → E) :
    Orthonormal 𝕜 fun i : { i | gramSchmidtNormed 𝕜 f i ≠ 0 } => gramSchmidtNormed 𝕜 f i := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    ⊢ Orthonormal 𝕜 fun i => gramSchmidtNormed 𝕜 f ↑i
  -/
  refine ⟨fun i => gramSchmidtNormed_unit_length' i.prop, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    ⊢ Pairwise fun i j => Eq (Inner.inner ((fun i => gramSchmidtNormed 𝕜 f ↑i) i)  …
  -/
  rintro i j (hij : ¬_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    i j : ↑(setOf fun i => Ne (gramSchmidtNormed 𝕜 f i) 0)
    hij : Not (Eq i j)
    ⊢ Eq (Inner.inner ((fun i => gramSchmidtNormed 𝕜 f ↑i) i) ((fun i => gramSchmi …
  -/
  rw [Subtype.ext_iff] at hij
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    i j : ↑(setOf fun i => Ne (gramSchmidtNormed 𝕜 f i) 0)
    hij : Not (Eq ↑i ↑j)
    ⊢ Eq (Inner.inner ((fun i => gramSchmidtNormed 𝕜 f ↑i) i) ((fun i => gramSchmi …
  -/
  simp [gramSchmidtNormed, inner_smul_left, inner_smul_right, gramSchmidt_orthogonal 𝕜 f hij]
  /-
    🎉 no goals
  -/


theorem span_gramSchmidtNormed (f : ι → E) (s : Set ι) :
    span 𝕜 (gramSchmidtNormed 𝕜 f '' s) = span 𝕜 (gramSchmidt 𝕜 f '' s) := by
  refine span_eq_span
    (Set.image_subset_iff.2 fun i hi => smul_mem _ _ <| subset_span <| mem_image_of_mem _ hi)
    (Set.image_subset_iff.2 fun i hi =>
      span_mono (image_subset _ <| singleton_subset_set_iff.2 hi) ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    s : Set ι
    i : ι
    hi : Membership.mem s i
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image (gramSchmidtNormed 𝕜 f) ↑(Single …
  -/
  simp only [coe_singleton, Set.image_singleton]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    s : Set ι
    i : ι
    hi : Membership.mem s i
    ⊢ Membership.mem (Submodule.span 𝕜 (Singleton.singleton (gramSchmidtNormed 𝕜 f …
  -/
  by_cases h : gramSchmidt 𝕜 f i = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      s : Set ι
      i : ι
      hi : Membership.mem s i
      h : Eq (gramSchmidt 𝕜 f i) 0
      ⊢ Membership.mem (Submodule.span 𝕜 (Singleton.singleton (gramSchmidtNormed 𝕜 f …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      s : Set ι
      i : ι
      hi : Membership.mem s i
      h : Not (Eq (gramSchmidt 𝕜 f i) 0)
      ⊢ Membership.mem (Submodule.span 𝕜 (Singleton.singleton (gramSchmidtNormed 𝕜 f …
    -/
  · refine mem_span_singleton.2 ⟨‖gramSchmidt 𝕜 f i‖, smul_inv_smul₀ ?_ _⟩
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝² : LinearOrder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      inst✝ : WellFoundedLT ι
      f : ι → E
      s : Set ι
      i : ι
      hi : Membership.mem s i
      h : Not (Eq (gramSchmidt 𝕜 f i) 0)
      ⊢ Ne (↑(Norm.norm (gramSchmidt 𝕜 f i))) 0
    -/
    exact mod_cast norm_ne_zero_iff.2 h
    /-
      🎉 no goals
    -/


theorem span_gramSchmidtNormed_range (f : ι → E) :
    span 𝕜 (range (gramSchmidtNormed 𝕜 f)) = span 𝕜 (range (gramSchmidt 𝕜 f)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝² : LinearOrder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    inst✝ : WellFoundedLT ι
    f : ι → E
    ⊢ Eq (Submodule.span 𝕜 (Set.range (gramSchmidtNormed 𝕜 f))) (Submodule.span 𝕜  …
  -/
  simpa only [image_univ.symm] using span_gramSchmidtNormed f univ
  /-
    🎉 no goals
  -/


/-- Given an indexed family `f : ι → E` of vectors in an inner product space `E`, for which the
size of the index set is the dimension of `E`, produce an orthonormal basis for `E` which agrees
with the orthonormal set produced by the Gram-Schmidt orthonormalization process on the elements of
`ι` for which this process gives a nonzero number. -/
noncomputable def gramSchmidtOrthonormalBasis : OrthonormalBasis ι 𝕜 E :=
  ((gramSchmidt_orthonormal' f).exists_orthonormalBasis_extension_of_card_eq
    (v := gramSchmidtNormed 𝕜 f) h).choose


theorem gramSchmidtOrthonormalBasis_apply {f : ι → E} {i : ι} (hi : gramSchmidtNormed 𝕜 f i ≠ 0) :
    gramSchmidtOrthonormalBasis h f i = gramSchmidtNormed 𝕜 f i :=
  ((gramSchmidt_orthonormal' f).exists_orthonormalBasis_extension_of_card_eq
    (v := gramSchmidtNormed 𝕜 f) h).choose_spec i hi


theorem gramSchmidtOrthonormalBasis_apply_of_orthogonal {f : ι → E}
    (hf : Pairwise fun i j => ⟪f i, f j⟫ = 0) {i : ι} (hi : f i ≠ 0) :
    gramSchmidtOrthonormalBasis h f i = (‖f i‖⁻¹ : 𝕜) • f i := by
  have H : gramSchmidtNormed 𝕜 f i = (‖f i‖⁻¹ : 𝕜) • f i := by
    rw [gramSchmidtNormed, gramSchmidt_of_orthogonal 𝕜 hf]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
    i : ι
    hi : Ne (f i) 0
    H : Eq (gramSchmidtNormed 𝕜 f i) (HSMul.hSMul (Inv.inv ↑(Norm.norm (f i))) (f  …
    ⊢ Eq ((gramSchmidtOrthonormalBasis h f) i) (HSMul.hSMul (Inv.inv ↑(Norm.norm ( …
  -/
  rw [gramSchmidtOrthonormalBasis_apply h, H]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    hf : Pairwise fun i j => Eq (Inner.inner (f i) (f j)) 0
    i : ι
    hi : Ne (f i) 0
    H : Eq (gramSchmidtNormed 𝕜 f i) (HSMul.hSMul (Inv.inv ↑(Norm.norm (f i))) (f  …
    ⊢ Ne (gramSchmidtNormed 𝕜 f i) 0
  -/
  simpa [H] using hi
  /-
    🎉 no goals
  -/


theorem inner_gramSchmidtOrthonormalBasis_eq_zero {f : ι → E} {i : ι}
    (hi : gramSchmidtNormed 𝕜 f i = 0) (j : ι) : ⟪gramSchmidtOrthonormalBasis h f i, f j⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i : ι
    hi : Eq (gramSchmidtNormed 𝕜 f i) 0
    j : ι
    ⊢ Eq (Inner.inner ((gramSchmidtOrthonormalBasis h f) i) (f j)) 0
  -/
  rw [← mem_orthogonal_singleton_iff_inner_right]
  suffices span 𝕜 (gramSchmidtNormed 𝕜 f '' Set.Iic j) ⟂ 𝕜 ∙ gramSchmidtOrthonormalBasis h f i by
    apply this
    rw [span_gramSchmidtNormed]
    exact mem_span_gramSchmidt 𝕜 f le_rfl
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i : ι
    hi : Eq (gramSchmidtNormed 𝕜 f i) 0
    j : ι
    ⊢ (Submodule.span 𝕜 (Set.image (gramSchmidtNormed 𝕜 f) (Set.Iic j))).IsOrtho ( …
  -/
  rw [isOrtho_span]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i : ι
    hi : Eq (gramSchmidtNormed 𝕜 f i) 0
    j : ι
    ⊢ ∀ ⦃u : E⦄, Membership.mem (Set.image (gramSchmidtNormed 𝕜 f) (Set.Iic j)) u  …
  -/
  rintro u ⟨k, _, rfl⟩ v (rfl : v = _)
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i : ι
    hi : Eq (gramSchmidtNormed 𝕜 f i) 0
    j k : ι
    left✝ : Membership.mem (Set.Iic j) k
    ⊢ Eq (Inner.inner (gramSchmidtNormed 𝕜 f k) ((gramSchmidtOrthonormalBasis h f) …
  -/
  by_cases hk : gramSchmidtNormed 𝕜 f k = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝⁴ : LinearOrder ι
      inst✝³ : LocallyFiniteOrderBot ι
      inst✝² : WellFoundedLT ι
      inst✝¹ : Fintype ι
      inst✝ : FiniteDimensional 𝕜 E
      h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
      f : ι → E
      i : ι
      hi : Eq (gramSchmidtNormed 𝕜 f i) 0
      j k : ι
      left✝ : Membership.mem (Set.Iic j) k
      hk : Eq (gramSchmidtNormed 𝕜 f k) 0
      ⊢ Eq (Inner.inner (gramSchmidtNormed 𝕜 f k) ((gramSchmidtOrthonormalBasis h f) …
    -/
  · rw [hk, inner_zero_left]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i : ι
    hi : Eq (gramSchmidtNormed 𝕜 f i) 0
    j k : ι
    left✝ : Membership.mem (Set.Iic j) k
    hk : Not (Eq (gramSchmidtNormed 𝕜 f k) 0)
    ⊢ Eq (Inner.inner (gramSchmidtNormed 𝕜 f k) ((gramSchmidtOrthonormalBasis h f) …
  -/
  rw [← gramSchmidtOrthonormalBasis_apply h hk]
  have : k ≠ i := by
    rintro rfl
    exact hk hi
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i : ι
    hi : Eq (gramSchmidtNormed 𝕜 f i) 0
    j k : ι
    left✝ : Membership.mem (Set.Iic j) k
    hk : Not (Eq (gramSchmidtNormed 𝕜 f k) 0)
    this : Ne k i
    ⊢ Eq (Inner.inner ((gramSchmidtOrthonormalBasis h f) k) ((gramSchmidtOrthonorm …
  -/
  exact (gramSchmidtOrthonormalBasis h f).orthonormal.2 this
  /-
    🎉 no goals
  -/


theorem gramSchmidtOrthonormalBasis_inv_triangular {i j : ι} (hij : i < j) :
    ⟪gramSchmidtOrthonormalBasis h f j, f i⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i j : ι
    hij : LT.lt i j
    ⊢ Eq (Inner.inner ((gramSchmidtOrthonormalBasis h f) j) (f i)) 0
  -/
  by_cases hi : gramSchmidtNormed 𝕜 f j = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : InnerProductSpace 𝕜 E
      ι : Type u_3
      inst✝⁴ : LinearOrder ι
      inst✝³ : LocallyFiniteOrderBot ι
      inst✝² : WellFoundedLT ι
      inst✝¹ : Fintype ι
      inst✝ : FiniteDimensional 𝕜 E
      h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
      f : ι → E
      i j : ι
      hij : LT.lt i j
      hi : Eq (gramSchmidtNormed 𝕜 f j) 0
      ⊢ Eq (Inner.inner ((gramSchmidtOrthonormalBasis h f) j) (f i)) 0
    -/
  · rw [inner_gramSchmidtOrthonormalBasis_eq_zero h hi]
    /-
      🎉 no goals
    -/
  · simp [gramSchmidtOrthonormalBasis_apply h hi, gramSchmidtNormed, inner_smul_left,
      gramSchmidt_inv_triangular 𝕜 f hij]


theorem gramSchmidtOrthonormalBasis_inv_triangular' {i j : ι} (hij : i < j) :
    (gramSchmidtOrthonormalBasis h f).repr (f i) j = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : RCLike 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁴ : LinearOrder ι
    inst✝³ : LocallyFiniteOrderBot ι
    inst✝² : WellFoundedLT ι
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    i j : ι
    hij : LT.lt i j
    ⊢ Eq ((gramSchmidtOrthonormalBasis h f).repr (f i) j) 0
  -/
  simpa [OrthonormalBasis.repr_apply_apply] using gramSchmidtOrthonormalBasis_inv_triangular h f hij
  /-
    🎉 no goals
  -/


/-- Given an indexed family `f : ι → E` of vectors in an inner product space `E`, for which the
size of the index set is the dimension of `E`, the matrix of coefficients of `f` with respect to the
orthonormal basis `gramSchmidtOrthonormalBasis` constructed from `f` is upper-triangular. -/
theorem gramSchmidtOrthonormalBasis_inv_blockTriangular :
    ((gramSchmidtOrthonormalBasis h f).toBasis.toMatrix f).BlockTriangular id := fun _ _ =>
  gramSchmidtOrthonormalBasis_inv_triangular' h f

-- Porting note: added a `DecidableEq` argument to help with timeouts in
-- `Mathlib/Analysis/InnerProductSpace/Orientation.lean`

theorem gramSchmidtOrthonormalBasis_det [DecidableEq ι] :
    (gramSchmidtOrthonormalBasis h f).toBasis.det f =
      ∏ i, ⟪gramSchmidtOrthonormalBasis h f i, f i⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : LocallyFiniteOrderBot ι
    inst✝³ : WellFoundedLT ι
    inst✝² : Fintype ι
    inst✝¹ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    inst✝ : DecidableEq ι
    ⊢ Eq ((gramSchmidtOrthonormalBasis h f).toBasis.det f) (Finset.univ.prod fun i …
  -/
  convert Matrix.det_of_upperTriangular (gramSchmidtOrthonormalBasis_inv_blockTriangular h f)
  /-
    case h.e'_3.a
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace 𝕜 E
    ι : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : LocallyFiniteOrderBot ι
    inst✝³ : WellFoundedLT ι
    inst✝² : Fintype ι
    inst✝¹ : FiniteDimensional 𝕜 E
    h : Eq (Module.finrank 𝕜 E) (Fintype.card ι)
    f : ι → E
    inst✝ : DecidableEq ι
    x✝ : ι
    a✝ : Membership.mem Finset.univ x✝
    ⊢ Eq (Inner.inner ((gramSchmidtOrthonormalBasis h f) x✝) (f x✝)) ((gramSchmidt …
  -/
  exact ((gramSchmidtOrthonormalBasis h f).repr_apply_apply (f _) _).symm
  /-
    🎉 no goals
  -/


