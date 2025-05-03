@[to_additive]
theorem le_of_forall_lt_one_mul_le (h : ∀ ε < 1, a * ε ≤ b) : a ≤ b :=
  le_of_forall_one_lt_le_mul (α := αᵒᵈ) h


@[to_additive]
theorem le_of_forall_one_lt_div_le (h : ∀ ε : α, 1 < ε → a / ε ≤ b) : a ≤ b :=
  le_of_forall_lt_one_mul_le fun ε ε1 => by
    /-
      α : Type u_1
      inst✝³ : Group α
      inst✝² : LinearOrder α
      inst✝¹ : MulLeftMono α
      inst✝ : DenselyOrdered α
      a b : α
      h : ∀ (ε : α), LT.lt 1 ε → LE.le (HDiv.hDiv a ε) b
      ε : α
      ε1 : LT.lt ε 1
      ⊢ LE.le (HMul.hMul a ε) b
    -/
    simpa only [div_eq_mul_inv, inv_inv] using h ε⁻¹ (Left.one_lt_inv_iff.2 ε1)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem le_iff_forall_lt_one_mul_le : a ≤ b ↔ ∀ ε < 1, a * ε ≤ b :=
  le_iff_forall_one_lt_le_mul (α := αᵒᵈ)


@[to_additive]
private lemma exists_lt_mul_left [Group α] [LT α] [DenselyOrdered α]
    [CovariantClass α α (Function.swap (· * ·)) (· < ·)] {a b c : α} (hc : c < a * b) :
    ∃ a' < a, c < a' * b := by
  /-
    α : Type u_1
    inst✝³ : Group α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (Function.swap fun x1 x2 => HMul.hMul x1 x2) fun x1 …
    a b c : α
    hc : LT.lt c (HMul.hMul a b)
    ⊢ Exists fun a' => And (LT.lt a' a) (LT.lt c (HMul.hMul a' b))
  -/
  obtain ⟨a', hc', ha'⟩ := exists_between (div_lt_iff_lt_mul.2 hc)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : Group α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (Function.swap fun x1 x2 => HMul.hMul x1 x2) fun x1 …
    a b c : α
    hc : LT.lt c (HMul.hMul a b)
    a' : α
    hc' : LT.lt (HDiv.hDiv c b) a'
    ha' : LT.lt a' a
    ⊢ Exists fun a' => And (LT.lt a' a) (LT.lt c (HMul.hMul a' b))
  -/
  exact ⟨a', ha', div_lt_iff_lt_mul.1 hc'⟩
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma exists_lt_mul_right [CommGroup α] [LT α] [DenselyOrdered α]
    [CovariantClass α α (· * ·) (· < ·)] {a b c : α} (hc : c < a * b) :
    ∃ b' < b, c < a * b' := by
  /-
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LT.lt x …
    a b c : α
    hc : LT.lt c (HMul.hMul a b)
    ⊢ Exists fun b' => And (LT.lt b' b) (LT.lt c (HMul.hMul a b'))
  -/
  obtain ⟨a', hc', ha'⟩ := exists_between (div_lt_iff_lt_mul'.2 hc)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LT.lt x …
    a b c : α
    hc : LT.lt c (HMul.hMul a b)
    a' : α
    hc' : LT.lt (HDiv.hDiv c a) a'
    ha' : LT.lt a' b
    ⊢ Exists fun b' => And (LT.lt b' b) (LT.lt c (HMul.hMul a b'))
  -/
  exact ⟨a', ha', div_lt_iff_lt_mul'.1 hc'⟩
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma exists_mul_left_lt [Group α] [LT α] [DenselyOrdered α]
    [CovariantClass α α (Function.swap (· * ·)) (· < ·)] {a b c : α} (hc : a * b < c) :
    ∃ a' > a, a' * b < c := by
  /-
    α : Type u_1
    inst✝³ : Group α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (Function.swap fun x1 x2 => HMul.hMul x1 x2) fun x1 …
    a b c : α
    hc : LT.lt (HMul.hMul a b) c
    ⊢ Exists fun a' => And (GT.gt a' a) (LT.lt (HMul.hMul a' b) c)
  -/
  obtain ⟨a', ha', hc'⟩ := exists_between (lt_div_iff_mul_lt.2 hc)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : Group α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (Function.swap fun x1 x2 => HMul.hMul x1 x2) fun x1 …
    a b c : α
    hc : LT.lt (HMul.hMul a b) c
    a' : α
    ha' : LT.lt a a'
    hc' : LT.lt a' (HDiv.hDiv c b)
    ⊢ Exists fun a' => And (GT.gt a' a) (LT.lt (HMul.hMul a' b) c)
  -/
  exact ⟨a', ha', lt_div_iff_mul_lt.1 hc'⟩
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma exists_mul_right_lt [CommGroup α] [LT α] [DenselyOrdered α]
    [CovariantClass α α (· * ·) (· < ·)] {a b c : α} (hc : a * b < c) :
    ∃ b' > b, a * b' < c := by
  /-
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LT.lt x …
    a b c : α
    hc : LT.lt (HMul.hMul a b) c
    ⊢ Exists fun b' => And (GT.gt b' b) (LT.lt (HMul.hMul a b') c)
  -/
  obtain ⟨a', ha', hc'⟩ := exists_between (lt_div_iff_mul_lt'.2 hc)
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LT α
    inst✝¹ : DenselyOrdered α
    inst✝ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LT.lt x …
    a b c : α
    hc : LT.lt (HMul.hMul a b) c
    a' : α
    ha' : LT.lt b a'
    hc' : LT.lt a' (HDiv.hDiv c a)
    ⊢ Exists fun b' => And (GT.gt b' b) (LT.lt (HMul.hMul a b') c)
  -/
  exact ⟨a', ha', lt_div_iff_mul_lt'.1 hc'⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma le_mul_of_forall_lt [CommGroup α] [LinearOrder α] [CovariantClass α α (· * ·) (· ≤ ·)]
    [DenselyOrdered α] {a b c : α} (h : ∀ a' > a, ∀ b' > b, c ≤ a' * b') :
    c ≤ a * b := by
  /-
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), GT.gt a' a → ∀ (b' : α), GT.gt b' b → LE.le c (HMul.hMul a' b')
    ⊢ LE.le c (HMul.hMul a b)
  -/
  refine le_of_forall_le_of_dense fun d hd ↦ ?_
  /-
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), GT.gt a' a → ∀ (b' : α), GT.gt b' b → LE.le c (HMul.hMul a' b')
    d : α
    hd : LT.lt (HMul.hMul a b) d
    ⊢ LE.le c d
  -/
  obtain ⟨a', ha', hd⟩ := exists_mul_left_lt hd
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), GT.gt a' a → ∀ (b' : α), GT.gt b' b → LE.le c (HMul.hMul a' b')
    d : α
    hd✝ : LT.lt (HMul.hMul a b) d
    a' : α
    ha' : GT.gt a' a
    hd : LT.lt (HMul.hMul a' b) d
    ⊢ LE.le c d
  -/
  obtain ⟨b', hb', hd⟩ := exists_mul_right_lt hd
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), GT.gt a' a → ∀ (b' : α), GT.gt b' b → LE.le c (HMul.hMul a' b')
    d : α
    hd✝¹ : LT.lt (HMul.hMul a b) d
    a' : α
    ha' : GT.gt a' a
    hd✝ : LT.lt (HMul.hMul a' b) d
    b' : α
    hb' : GT.gt b' b
    hd : LT.lt (HMul.hMul a' b') d
    ⊢ LE.le c d
  -/
  exact (h a' ha' b' hb').trans hd.le
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mul_le_of_forall_lt [CommGroup α] [LinearOrder α] [CovariantClass α α (· * ·) (· ≤ ·)]
    [DenselyOrdered α] {a b c : α} (h : ∀ a' < a, ∀ b' < b, a' * b' ≤ c) :
    a * b ≤ c := by
  /-
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), LT.lt a' a → ∀ (b' : α), LT.lt b' b → LE.le (HMul.hMul a' b') c
    ⊢ LE.le (HMul.hMul a b) c
  -/
  refine le_of_forall_ge_of_dense fun d hd ↦ ?_
  /-
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), LT.lt a' a → ∀ (b' : α), LT.lt b' b → LE.le (HMul.hMul a' b') c
    d : α
    hd : LT.lt d (HMul.hMul a b)
    ⊢ LE.le d c
  -/
  obtain ⟨a', ha', hd⟩ := exists_lt_mul_left hd
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), LT.lt a' a → ∀ (b' : α), LT.lt b' b → LE.le (HMul.hMul a' b') c
    d : α
    hd✝ : LT.lt d (HMul.hMul a b)
    a' : α
    ha' : LT.lt a' a
    hd : LT.lt d (HMul.hMul a' b)
    ⊢ LE.le d c
  -/
  obtain ⟨b', hb', hd⟩ := exists_lt_mul_right hd
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : CommGroup α
    inst✝² : LinearOrder α
    inst✝¹ : CovariantClass α α (fun x1 x2 => HMul.hMul x1 x2) fun x1 x2 => LE.le  …
    inst✝ : DenselyOrdered α
    a b c : α
    h : ∀ (a' : α), LT.lt a' a → ∀ (b' : α), LT.lt b' b → LE.le (HMul.hMul a' b') c
    d : α
    hd✝¹ : LT.lt d (HMul.hMul a b)
    a' : α
    ha' : LT.lt a' a
    hd✝ : LT.lt d (HMul.hMul a' b)
    b' : α
    hb' : LT.lt b' b
    hd : LT.lt d (HMul.hMul a' b')
    ⊢ LE.le d c
  -/
  exact hd.le.trans (h a' ha' b' hb')
  /-
    🎉 no goals
  -/


