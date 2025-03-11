/-- A nonarchimedean function satisfies the triangle inequality. -/
theorem add_le {α : Type*} [Add α] {f : α → ℝ} (hf : ∀ x : α, 0 ≤ f x)
    (hna : IsNonarchimedean f) {a b : α} : f (a + b) ≤ f a + f b := by
  /-
    α : Type u_1
    inst✝ : Add α
    f : α → Real
    hf : ∀ (x : α), LE.le 0 (f x)
    hna : IsNonarchimedean f
    a b : α
    ⊢ LE.le (f (HAdd.hAdd a b)) (HAdd.hAdd (f a) (f b))
  -/
  apply le_trans (hna _ _)
  /-
    α : Type u_1
    inst✝ : Add α
    f : α → Real
    hf : ∀ (x : α), LE.le 0 (f x)
    hna : IsNonarchimedean f
    a b : α
    ⊢ LE.le (Max.max (f a) (f b)) (HAdd.hAdd (f a) (f b))
  -/
  rw [max_le_iff, le_add_iff_nonneg_right, le_add_iff_nonneg_left]
  /-
    α : Type u_1
    inst✝ : Add α
    f : α → Real
    hf : ∀ (x : α), LE.le 0 (f x)
    hna : IsNonarchimedean f
    a b : α
    ⊢ And (LE.le 0 (f b)) (LE.le 0 (f a))
  -/
  exact ⟨hf _, hf _⟩
  /-
    🎉 no goals
  -/


/-- If `f` is a nonarchimedean additive group seminorm on `α`, then for every `n : ℕ` and `a : α`,
  we have `f (n • a) ≤ (f a)`. -/
theorem nsmul_le {F α : Type*} [AddGroup α] [FunLike F α ℝ]
    [AddGroupSeminormClass F α ℝ] {f : F} (hna : IsNonarchimedean f) {n : ℕ} {a : α} :
    f (n • a) ≤ f a := by
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a : α
    ⊢ LE.le (f (HSMul.hSMul n a)) (f a)
  -/
  let _ := AddGroupSeminormClass.toSeminormedAddGroup f
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a : α
    x✝ : SeminormedAddGroup α := AddGroupSeminormClass.toSeminormedAddGroup f
    ⊢ LE.le (f (HSMul.hSMul n a)) (f a)
  -/
  have := AddGroupSeminormClass.isUltrametricDist hna
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a : α
    x✝ : SeminormedAddGroup α := AddGroupSeminormClass.toSeminormedAddGroup f
    this : IsUltrametricDist α
    ⊢ LE.le (f (HSMul.hSMul n a)) (f a)
  -/
  simp only [← AddGroupSeminormClass.toSeminormedAddGroup_norm_eq]
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a : α
    x✝ : SeminormedAddGroup α := AddGroupSeminormClass.toSeminormedAddGroup f
    this : IsUltrametricDist α
    ⊢ LE.le (Norm.norm (HSMul.hSMul n a)) (Norm.norm a)
  -/
  exact norm_nsmul_le _ _
  /-
    🎉 no goals
  -/


/-- If `f` is a nonarchimedean additive group seminorm on `α`, then for every `n : ℕ` and `a : α`,
  we have `f (n * a) ≤ (f a)`. -/
theorem nmul_le {F α : Type*} [Ring α] [FunLike F α ℝ] [AddGroupSeminormClass F α ℝ]
    {f : F} (hna : IsNonarchimedean f) {n : ℕ} {a : α} : f (n * a) ≤ f a := by
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a : α
    ⊢ LE.le (f (HMul.hMul (↑n) a)) (f a)
  -/
  rw [← nsmul_eq_mul]
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : Ring α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a : α
    ⊢ LE.le (f (HSMul.hSMul n a)) (f a)
  -/
  exact nsmul_le hna
  /-
    🎉 no goals
  -/


/-- If `f` is a nonarchimedean additive group seminorm on `α` and `x y : α` are such that
  `f x ≠ f y`, then `f (x + y) = max (f x) (f y)`. -/
theorem add_eq_max_of_ne {F α : Type*} [AddGroup α] [FunLike F α ℝ]
    [AddGroupSeminormClass F α ℝ] {f : F} (hna : IsNonarchimedean f) {x y : α} (hne : f x ≠ f y) :
    f (x + y) = max (f x) (f y) := by
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    x y : α
    hne : Ne (f x) (f y)
    ⊢ Eq (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
  -/
  let _ := AddGroupSeminormClass.toSeminormedAddGroup f
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    x y : α
    hne : Ne (f x) (f y)
    x✝ : SeminormedAddGroup α := AddGroupSeminormClass.toSeminormedAddGroup f
    ⊢ Eq (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
  -/
  have := AddGroupSeminormClass.isUltrametricDist hna
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    x y : α
    hne : Ne (f x) (f y)
    x✝ : SeminormedAddGroup α := AddGroupSeminormClass.toSeminormedAddGroup f
    this : IsUltrametricDist α
    ⊢ Eq (f (HAdd.hAdd x y)) (Max.max (f x) (f y))
  -/
  simp only [← AddGroupSeminormClass.toSeminormedAddGroup_norm_eq] at hne ⊢
  /-
    F : Type u_1
    α : Type u_2
    inst✝² : AddGroup α
    inst✝¹ : FunLike F α Real
    inst✝ : AddGroupSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    x y : α
    hne : Ne (Norm.norm x) (Norm.norm y)
    x✝ : SeminormedAddGroup α := AddGroupSeminormClass.toSeminormedAddGroup f
    this : IsUltrametricDist α
    ⊢ Eq (Norm.norm (HAdd.hAdd x y)) (Max.max (Norm.norm x) (Norm.norm y))
  -/
  exact norm_add_eq_max_of_norm_ne_norm hne
  /-
    🎉 no goals
  -/


/-- Given a nonarchimedean additive group seminorm `f` on `α`, a function `g : β → α` and a finset
  `t : Finset β`, we can always find `b : β`, belonging to `t` if `t` is nonempty, such that
  `f (t.sum g) ≤ f (g b)` . -/
theorem finset_image_add {F α β : Type*} [AddCommGroup α] [FunLike F α ℝ]
    [AddGroupSeminormClass F α ℝ] [Nonempty β] {f : F} (hna : IsNonarchimedean f)
    (g : β → α) (t : Finset β) :
    ∃ b : β, (t.Nonempty → b ∈ t) ∧ f (t.sum g) ≤ f (g b) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    t : Finset β
    ⊢ Exists fun b => And (t.Nonempty → Membership.mem t b) (LE.le (f (t.sum g)) ( …
  -/
  let _ := AddGroupSeminormClass.toSeminormedAddCommGroup f
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    t : Finset β
    x✝ : SeminormedAddCommGroup α := AddGroupSeminormClass.toSeminormedAddCommGrou …
    ⊢ Exists fun b => And (t.Nonempty → Membership.mem t b) (LE.le (f (t.sum g)) ( …
  -/
  have := AddGroupSeminormClass.isUltrametricDist hna
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    t : Finset β
    x✝ : SeminormedAddCommGroup α := AddGroupSeminormClass.toSeminormedAddCommGrou …
    this : IsUltrametricDist α
    ⊢ Exists fun b => And (t.Nonempty → Membership.mem t b) (LE.le (f (t.sum g)) ( …
  -/
  simp only [← AddGroupSeminormClass.toSeminormedAddCommGroup_norm_eq]
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    t : Finset β
    x✝ : SeminormedAddCommGroup α := AddGroupSeminormClass.toSeminormedAddCommGrou …
    this : IsUltrametricDist α
    ⊢ Exists fun b => And (t.Nonempty → Membership.mem t b) (LE.le (Norm.norm (t.s …
  -/
  apply exists_norm_finset_sum_le
  /-
    🎉 no goals
  -/


/-- Given a nonarchimedean additive group seminorm `f` on `α`, a function `g : β → α` and a
  nonempty finset `t : Finset β`, we can always find `b : β` belonging to `t` such that
  `f (t.sum g) ≤ f (g b)` . -/
theorem finset_image_add_of_nonempty {F α β : Type*} [AddCommGroup α] [FunLike F α ℝ]
    [AddGroupSeminormClass F α ℝ] [Nonempty β] {f : F} (hna : IsNonarchimedean f)
    (g : β → α) {t : Finset β} (ht : t.Nonempty) :
    ∃ b : β, (b ∈ t) ∧ f (t.sum g) ≤ f (g b) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    t : Finset β
    ht : t.Nonempty
    ⊢ Exists fun b => And (Membership.mem t b) (LE.le (f (t.sum g)) (f (g b)))
  -/
  obtain ⟨b, hbt, hbf⟩ := finset_image_add hna g t
  /-
    case intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    t : Finset β
    ht : t.Nonempty
    b : β
    hbt : t.Nonempty → Membership.mem t b
    hbf : LE.le (f (t.sum g)) (f (g b))
    ⊢ Exists fun b => And (Membership.mem t b) (LE.le (f (t.sum g)) (f (g b)))
  -/
  exact ⟨b, hbt ht, hbf⟩
  /-
    🎉 no goals
  -/


/-- Given a nonarchimedean additive group seminorm `f` on `α`, a function `g : β → α` and a
  multiset `s : Multiset β`, we can always find `b : β`, belonging to `s` if `s` is nonempty,
  such that `f (t.sum g) ≤ f (g b)` . -/
theorem multiset_image_add {F α β : Type*} [AddCommGroup α] [FunLike F α ℝ]
    [AddGroupSeminormClass F α ℝ] [Nonempty β] {f : F} (hna : IsNonarchimedean f)
    (g : β → α) (s : Multiset β) :
    ∃ b : β, (s ≠ 0 → b ∈ s) ∧ f (Multiset.map g s).sum ≤ f (g b) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    s : Multiset β
    ⊢ Exists fun b => And (Ne s 0 → Membership.mem s b) (LE.le (f (Multiset.map g  …
  -/
  let _ := AddGroupSeminormClass.toSeminormedAddCommGroup f
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    s : Multiset β
    x✝ : SeminormedAddCommGroup α := AddGroupSeminormClass.toSeminormedAddCommGrou …
    ⊢ Exists fun b => And (Ne s 0 → Membership.mem s b) (LE.le (f (Multiset.map g  …
  -/
  have := AddGroupSeminormClass.isUltrametricDist hna
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    s : Multiset β
    x✝ : SeminormedAddCommGroup α := AddGroupSeminormClass.toSeminormedAddCommGrou …
    this : IsUltrametricDist α
    ⊢ Exists fun b => And (Ne s 0 → Membership.mem s b) (LE.le (f (Multiset.map g  …
  -/
  simp only [← AddGroupSeminormClass.toSeminormedAddCommGroup_norm_eq]
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    s : Multiset β
    x✝ : SeminormedAddCommGroup α := AddGroupSeminormClass.toSeminormedAddCommGrou …
    this : IsUltrametricDist α
    ⊢ Exists fun b => And (Ne s 0 → Membership.mem s b) (LE.le (Norm.norm (Multise …
  -/
  apply exists_norm_multiset_sum_le
  /-
    🎉 no goals
  -/


/-- Given a nonarchimedean additive group seminorm `f` on `α`, a function `g : β → α` and a
  nonempty multiset `s : Multiset β`, we can always find `b : β` belonging to `s` such that
  `f (t.sum g) ≤ f (g b)` . -/
theorem multiset_image_add_of_nonempty {F α β : Type*} [AddCommGroup α] [FunLike F α ℝ]
    [AddGroupSeminormClass F α ℝ] [Nonempty β] {f : F} (hna : IsNonarchimedean f)
    (g : β → α) {s : Multiset β} (hs : s ≠ 0) :
    ∃ b : β, (b ∈ s) ∧ f (Multiset.map g s).sum ≤ f (g b) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    s : Multiset β
    hs : Ne s 0
    ⊢ Exists fun b => And (Membership.mem s b) (LE.le (f (Multiset.map g s).sum) ( …
  -/
  obtain ⟨b, hbs, hbf⟩ := multiset_image_add hna g s
  /-
    case intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : AddCommGroup α
    inst✝² : FunLike F α Real
    inst✝¹ : AddGroupSeminormClass F α Real
    inst✝ : Nonempty β
    f : F
    hna : IsNonarchimedean ⇑f
    g : β → α
    s : Multiset β
    hs : Ne s 0
    b : β
    hbs : Ne s 0 → Membership.mem s b
    hbf : LE.le (f (Multiset.map g s).sum) (f (g b))
    ⊢ Exists fun b => And (Membership.mem s b) (LE.le (f (Multiset.map g s).sum) ( …
  -/
  exact ⟨b, hbs hs, hbf⟩
  /-
    🎉 no goals
  -/


/-- If `f` is a nonarchimedean additive group seminorm on a commutative ring `α`, `n : ℕ`, and
  `a b : α`, then we can find `m : ℕ` such that `m ≤ n` and
  `f ((a + b) ^ n) ≤ (f (a ^ m)) * (f (b ^ (n - m)))`. -/
theorem add_pow_le {F α : Type*} [CommRing α] [FunLike F α ℝ]
    [RingSeminormClass F α ℝ] {f : F} (hna : IsNonarchimedean f) (n : ℕ) (a b : α) :
    ∃ m < n + 1, f ((a + b) ^ n) ≤ f (a ^ m) * f (b ^ (n - m)) := by
  obtain ⟨m, hm_lt, hM⟩ := finset_image_add hna
    (fun m => a ^ m * b ^ (n - m) * ↑(n.choose m)) (Finset.range (n + 1))
  simp only [Finset.nonempty_range_iff, ne_eq, Nat.succ_ne_zero, not_false_iff, Finset.mem_range,
    if_true, forall_true_left] at hm_lt
  /-
    case intro.intro
    F : Type u_1
    α : Type u_2
    inst✝² : CommRing α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a b : α
    m : Nat
    hM : LE.le (f ((Finset.range (HAdd.hAdd n 1)).sum fun m => HMul.hMul (HMul.hMu …
    hm_lt : LT.lt m (HAdd.hAdd n 1)
    ⊢ Exists fun m => And (LT.lt m (HAdd.hAdd n 1)) (LE.le (f (HPow.hPow (HAdd.hAd …
  -/
  refine ⟨m, hm_lt, ?_⟩
  /-
    case intro.intro
    F : Type u_1
    α : Type u_2
    inst✝² : CommRing α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a b : α
    m : Nat
    hM : LE.le (f ((Finset.range (HAdd.hAdd n 1)).sum fun m => HMul.hMul (HMul.hMu …
    hm_lt : LT.lt m (HAdd.hAdd n 1)
    ⊢ LE.le (f (HPow.hPow (HAdd.hAdd a b) n)) (HMul.hMul (f (HPow.hPow a m)) (f (H …
  -/
  simp only [← add_pow] at hM
  /-
    case intro.intro
    F : Type u_1
    α : Type u_2
    inst✝² : CommRing α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a b : α
    m : Nat
    hm_lt : LT.lt m (HAdd.hAdd n 1)
    hM : LE.le (f (HPow.hPow (HAdd.hAdd a b) n)) (f (HMul.hMul (HMul.hMul (HPow.hP …
    ⊢ LE.le (f (HPow.hPow (HAdd.hAdd a b) n)) (HMul.hMul (f (HPow.hPow a m)) (f (H …
  -/
  rw [mul_comm] at hM
  /-
    case intro.intro
    F : Type u_1
    α : Type u_2
    inst✝² : CommRing α
    inst✝¹ : FunLike F α Real
    inst✝ : RingSeminormClass F α Real
    f : F
    hna : IsNonarchimedean ⇑f
    n : Nat
    a b : α
    m : Nat
    hm_lt : LT.lt m (HAdd.hAdd n 1)
    hM : LE.le (f (HPow.hPow (HAdd.hAdd a b) n)) (f (HMul.hMul (↑(n.choose m)) (HM …
    ⊢ LE.le (f (HPow.hPow (HAdd.hAdd a b) n)) (HMul.hMul (f (HPow.hPow a m)) (f (H …
  -/
  exact le_trans hM (le_trans (nmul_le hna) (map_mul_le_mul _ _ _))
  /-
    🎉 no goals
  -/


