instance : CharP Nimber 2 := by
  /-
    a b c : Nimber
    ⊢ CharP Nimber 2
  -/
  apply CharTwo.of_one_ne_zero_of_two_eq_zero one_ne_zero
  /-
    a b c : Nimber
    ⊢ Eq 2 0
  -/
  rw [← one_add_one_eq_two, add_self]
  /-
    🎉 no goals
  -/


private theorem two_zsmul (x : Nimber) : (2 : ℤ) • x = 0 := by
  /-
    x : Nimber
    ⊢ Eq (HSMul.hSMul 2 x) 0
  -/
  rw [_root_.two_zsmul]
  /-
    x : Nimber
    ⊢ Eq (HAdd.hAdd x x) 0
  -/
  exact add_self x
  /-
    🎉 no goals
  -/


private theorem add_eq_iff_eq_add : a + b = c ↔ a = c + b :=
  sub_eq_iff_eq_add


/-- Nimber multiplication is recursively defined so that `a * b` is the smallest nimber not equal to
`a' * b + a * b' + a' * b'` for `a' < a` and `b' < b`. -/
-- We write the binders like this so that the termination checker works.
protected def mul (a b : Nimber.{u}) : Nimber.{u} :=
  sInf {x | ∃ a', ∃ (_ : a' < a), ∃ b', ∃ (_ : b' < b),
    Nimber.mul a' b + Nimber.mul a b' + Nimber.mul a' b' = x}ᶜ
termination_by (a, b)


instance : Mul Nimber :=
  ⟨Nimber.mul⟩


theorem mul_def (a b : Nimber) :
    a * b = sInf {x | ∃ a' < a, ∃ b' < b, a' * b + a * b' + a' * b' = x}ᶜ := by
  /-
    a b : Nimber
    ⊢ Eq (HMul.hMul a b) (InfSet.sInf (HasCompl.compl (setOf fun x => Exists fun a …
  -/
  change Nimber.mul a b = _
  /-
    a b : Nimber
    ⊢ Eq (a.mul b) (InfSet.sInf (HasCompl.compl (setOf fun x => Exists fun a' => A …
  -/
  rw [Nimber.mul]
  /-
    a b : Nimber
    ⊢ Eq (InfSet.sInf (HasCompl.compl (setOf fun x => Exists fun a' => Exists fun  …
  -/
  simp_rw [exists_prop]
  /-
    a b : Nimber
    ⊢ Eq (InfSet.sInf (HasCompl.compl (setOf fun x => Exists fun a' => And (LT.lt  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The set in the definition of `Nimber.mul` is nonempty. -/
private theorem mul_nonempty (a b : Nimber.{u}) :
    {x | ∃ a' < a, ∃ b' < b, a' * b + a * b' + a' * b' = x}ᶜ.Nonempty := by
  convert nonempty_of_not_bddAbove <| not_bddAbove_compl_of_small
    ((fun x ↦ x.1 * b + a * x.2 + x.1 * x.2) '' Set.Iio a ×ˢ Set.Iio b)
  /-
    case h.e'_2.h.e'_3
    a b : Nimber
    ⊢ Eq (setOf fun x => Exists fun a' => And (LT.lt a' a) (Exists fun b' => And ( …
  -/
  ext
  /-
    case h.e'_2.h.e'_3.h
    a b x✝ : Nimber
    ⊢ Iff (Membership.mem (setOf fun x => Exists fun a' => And (LT.lt a' a) (Exist …
  -/
  simp_rw [Set.mem_setOf_eq, Set.mem_image, Set.mem_prod, Set.mem_Iio, Prod.exists]
  /-
    case h.e'_2.h.e'_3.h
    a b x✝ : Nimber
    ⊢ Iff (Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (E …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem exists_of_lt_mul (h : c < a * b) : ∃ a' < a, ∃ b' < b, a' * b + a * b' + a' * b' = c := by
  /-
    a b c : Nimber
    h : LT.lt c (HMul.hMul a b)
    ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Eq (HA …
  -/
  rw [mul_def] at h
  /-
    a b c : Nimber
    h : LT.lt c (InfSet.sInf (HasCompl.compl (setOf fun x => Exists fun a' => And  …
    ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Eq (HA …
  -/
  have := not_mem_of_lt_csInf' h
  /-
    a b c : Nimber
    h : LT.lt c (InfSet.sInf (HasCompl.compl (setOf fun x => Exists fun a' => And  …
    this : Not (Membership.mem (HasCompl.compl (setOf fun x => Exists fun a' => An …
    ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Eq (HA …
  -/
  rwa [Set.not_mem_compl_iff] at this
  /-
    🎉 no goals
  -/


theorem mul_le_of_forall_ne (h : ∀ a' < a, ∀ b' < b, a' * b + a * b' + a' * b' ≠ c) :
    a * b ≤ c := by
  /-
    a b c : Nimber
    h : ∀ (a' : Nimber), LT.lt a' a → ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd  …
    ⊢ LE.le (HMul.hMul a b) c
  -/
  by_contra! h'
  /-
    a b c : Nimber
    h : ∀ (a' : Nimber), LT.lt a' a → ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd  …
    h' : LT.lt c (HMul.hMul a b)
    ⊢ False
  -/
  have := exists_of_lt_mul h'
  /-
    a b c : Nimber
    h : ∀ (a' : Nimber), LT.lt a' a → ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd  …
    h' : LT.lt c (HMul.hMul a b)
    this : Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (E …
    ⊢ False
  -/
  tauto
  /-
    🎉 no goals
  -/


instance : MulZeroClass Nimber where
  mul_zero a := by
    /-
      a✝ b c : Nimber
      a : Nimber
      ⊢ Eq (HMul.hMul a 0) 0
    -/
    rw [← Nimber.le_zero]
    /-
      a✝ b c : Nimber
      a : Nimber
      ⊢ LE.le (HMul.hMul a 0) 0
    -/
    /-
      a✝ b c : Nimber
      a : Nimber
      ⊢ Eq (HMul.hMul 0 a) 0
    -/
    exact mul_le_of_forall_ne fun _ _ _ h ↦ (Nimber.not_lt_zero _ h).elim
    /-
      a✝ b c : Nimber
      a : Nimber
      ⊢ LE.le (HMul.hMul 0 a) 0
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  zero_mul a := by
    rw [← Nimber.le_zero]
    exact mul_le_of_forall_ne fun _ h ↦ (Nimber.not_lt_zero _ h).elim


private theorem mul_ne_of_lt : ∀ a' < a, ∀ b' < b, a' * b + a * b' + a' * b' ≠ a * b := by
  /-
    a b : Nimber
    ⊢ ∀ (a' : Nimber), LT.lt a' a → ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd (H …
  -/
  have H := csInf_mem (mul_nonempty a b)
  /-
    a b : Nimber
    H : Membership.mem (HasCompl.compl (setOf fun x => Exists fun a' => And (LT.lt …
    ⊢ ∀ (a' : Nimber), LT.lt a' a → ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd (H …
  -/
  rw [← mul_def] at H
  /-
    a b : Nimber
    H : Membership.mem (HasCompl.compl (setOf fun x => Exists fun a' => And (LT.lt …
    ⊢ ∀ (a' : Nimber), LT.lt a' a → ∀ (b' : Nimber), LT.lt b' b → Ne (HAdd.hAdd (H …
  -/
  simpa using H
  /-
    🎉 no goals
  -/


instance : NoZeroDivisors Nimber where
  eq_zero_or_eq_zero_of_mul_eq_zero {a b} h := by
    /-
      a✝ b✝ c : Nimber
      a b : Nimber
      h : Eq (HMul.hMul a b) 0
      ⊢ Or (Eq a 0) (Eq b 0)
    -/
    by_contra! hab
    /-
      a✝ b✝ c : Nimber
      a b : Nimber
      h : Eq (HMul.hMul a b) 0
      hab : And (Ne a 0) (Ne b 0)
      ⊢ False
    -/
    iterate 2 rw [← Nimber.pos_iff_ne_zero] at hab
    /-
      a✝ b✝ c : Nimber
      a b : Nimber
      h : Eq (HMul.hMul a b) 0
      hab : And (LT.lt 0 a) (LT.lt 0 b)
      ⊢ False
    -/
    apply (mul_ne_of_lt _ hab.1 _ hab.2).symm
    /-
      a✝ b✝ c : Nimber
      a b : Nimber
      h : Eq (HMul.hMul a b) 0
      hab : And (LT.lt 0 a) (LT.lt 0 b)
      ⊢ Eq (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 0 b) (HMul.hMul a 0)) (H …
    -/
    simpa only [zero_add, mul_zero, zero_mul]
    /-
      🎉 no goals
    -/


protected theorem mul_comm (a b : Nimber) : a * b = b * a := by
  /-
    a b : Nimber
    ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
  -/
  apply le_antisymm <;> refine mul_le_of_forall_ne fun x hx y hy ↦ ?_
  /-
    case a
    a b x : Nimber
    hx : LT.lt x a
    y : Nimber
    hy : LT.lt y b
    ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul x y)) ( …
  -/
  on_goal 1 => rw [add_comm (x * _), Nimber.mul_comm a, Nimber.mul_comm x, Nimber.mul_comm x]
  /-
    case a
    a b x : Nimber
    hx : LT.lt x a
    y : Nimber
    hy : LT.lt y b
    ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul y a) (HMul.hMul b x)) (HMul.hMul y x)) ( …
  -/
  on_goal 2 => rw [add_comm (x * _), ← Nimber.mul_comm y, ← Nimber.mul_comm a, ← Nimber.mul_comm y]
  /-
    case a
    a b x : Nimber
    hx : LT.lt x a
    y : Nimber
    hy : LT.lt y b
    ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul y a) (HMul.hMul b x)) (HMul.hMul y x)) ( …
  -/
  all_goals exact mul_ne_of_lt y hy x hx
  /-
    🎉 no goals
  -/
termination_by (a, b)


protected theorem mul_add (a b c : Nimber) : a * (b + c) = a * b + a * c := by
  /-
    a b c : Nimber
    ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
  -/
  apply le_antisymm
    /-
      case a
      a b c : Nimber
      ⊢ LE.le (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a  …
    -/
  · refine mul_le_of_forall_ne fun a' ha x hx ↦ ?_
    /-
      case a
      a b c a' : Nimber
      ha : LT.lt a' a
      x : Nimber
      hx : LT.lt x (HAdd.hAdd b c)
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' (HAdd.hAdd b c)) (HMul.hMul a x)) (HM …
    -/
    obtain (⟨b', h, rfl⟩ | ⟨c', h, rfl⟩) := exists_of_lt_add hx <;>
      /-
        case a.inl.intro.intro
        a b c a' : Nimber
        ha : LT.lt a' a
        b' : Nimber
        h : LT.lt b' b
        hx : LT.lt (HAdd.hAdd b' c) (HAdd.hAdd b c)
        ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' (HAdd.hAdd b c)) (HMul.hMul a (HAdd.h …
      -/
      rw [Nimber.mul_add a', Nimber.mul_add a, Nimber.mul_add a']
    /-
      case a.inl.intro.intro
      a b c a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      h : LT.lt b' b
      hx : LT.lt (HAdd.hAdd b' c) (HAdd.hAdd b c)
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a' c)) (HAdd …
    -/
    on_goal 1 => rw [← add_ne_add_left (a * c)]
    /-
      case a.inl.intro.intro
      a b c a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      h : LT.lt b' b
      hx : LT.lt (HAdd.hAdd b' c) (HAdd.hAdd b c)
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a …
    -/
    on_goal 2 => rw [← add_ne_add_left (a * b)]
    all_goals
      abel_nf
      simp only [two_zsmul, zero_add]
      rw [← add_assoc]
      exact mul_ne_of_lt _ ha _ h
    /-
      case a
      a b c : Nimber
      ⊢ LE.le (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c)) (HMul.hMul a (HAdd.hAdd b  …
    -/
  · apply add_le_of_forall_ne <;>
       /-
         case a.h₁
         a b c : Nimber
         ⊢ ∀ (a' : Nimber), LT.lt a' (HMul.hMul a b) → Ne (HAdd.hAdd a' (HMul.hMul a c) …
       -/
      (intro x' hx'; obtain ⟨x, hx, y, hy, rfl⟩ := exists_of_lt_mul hx')
      /-
        case a.h₁.intro.intro.intro.intro
        a b c x : Nimber
        hx : LT.lt x a
        y : Nimber
        hy : LT.lt y b
        hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
        ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
      -/
    · obtain h | h | h := lt_trichotomy (y + c) (b + c)
        /-
          case a.h₁.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y b
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd y c) (HAdd.hAdd b c)
          ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
        -/
      · have H := mul_ne_of_lt _ hx _ h
        /-
          case a.h₁.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y b
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd y c) (HAdd.hAdd b c)
          H : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (HAdd.hAdd b c)) (HMul.hMul a (HAdd. …
          ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
        -/
        rw [Nimber.mul_add x, Nimber.mul_add a, Nimber.mul_add x] at H
        /-
          case a.h₁.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y b
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd y c) (HAdd.hAdd b c)
          H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul x c)) (HAdd …
          ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
        -/
        abel_nf at H ⊢
        /-
          case a.h₁.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y b
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd y c) (HAdd.hAdd b c)
          H : Ne (HAdd.hAdd (HMul.hMul x b) (HAdd.hAdd (HSMul.hSMul 2 (HMul.hMul x c)) ( …
          ⊢ Not (Eq (HAdd.hAdd (HMul.hMul x b) (HAdd.hAdd (HMul.hMul a y) (HAdd.hAdd (HM …
        -/
        simpa only [two_zsmul, zero_add] using H
        /-
          🎉 no goals
        -/
        /-
          case a.h₁.intro.intro.intro.intro.inr.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y b
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
          h : Eq (HAdd.hAdd y c) (HAdd.hAdd b c)
          ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
        -/
      · exact (hy.ne <| add_left_injective _ h).elim
        /-
          🎉 no goals
        -/
        /-
          case a.h₁.intro.intro.intro.intro.inr.inr
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y b
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd y c)
          ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
        -/
      · obtain ⟨z, hz, hz'⟩ | ⟨c', hc, hc'⟩ := exists_of_lt_add h
          /-
            case a.h₁.intro.intro.intro.intro.inr.inr.inl.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y b
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd y c)
            z : Nimber
            hz : LT.lt z y
            hz' : Eq (HAdd.hAdd z c) (HAdd.hAdd b c)
            ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
          -/
        · exact ((hz.trans hy).ne <| add_left_injective _ hz').elim
          /-
            🎉 no goals
          -/
          /-
            case a.h₁.intro.intro.intro.intro.inr.inr.inr.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y b
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd y c)
            c' : Nimber
            hc : LT.lt c' c
            hc' : Eq (HAdd.hAdd y c') (HAdd.hAdd b c)
            ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
          -/
        · have := add_eq_iff_eq_add.1 hc'
          /-
            case a.h₁.intro.intro.intro.intro.inr.inr.inr.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y b
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd y c)
            c' : Nimber
            hc : LT.lt c' c
            hc' : Eq (HAdd.hAdd y c') (HAdd.hAdd b c)
            this : Eq y (HAdd.hAdd (HAdd.hAdd b c) c')
            ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.h …
          -/
          have H := mul_ne_of_lt _ hx _ hc
          rw [← hc', Nimber.mul_add a y c', ← add_ne_add_left (a * y), ← add_ne_add_left (a * c),
            ← add_ne_add_left (a * c'), ← add_eq_iff_eq_add.2 hc', Nimber.mul_add x,
            Nimber.mul_add x]
          /-
            case a.h₁.intro.intro.intro.intro.inr.inr.inr.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y b
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd y c)
            c' : Nimber
            hc : LT.lt c' c
            hc' : Eq (HAdd.hAdd y c') (HAdd.hAdd b c)
            this : Eq y (HAdd.hAdd (HAdd.hAdd b c) c')
            H : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a c')) (HMul.hMul x c' …
            ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.h …
          -/
          abel_nf at H ⊢
          /-
            case a.h₁.intro.intro.intro.intro.inr.inr.inr.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y b
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd y c)
            c' : Nimber
            hc : LT.lt c' c
            hc' : Eq (HAdd.hAdd y c') (HAdd.hAdd b c)
            this : Eq y (HAdd.hAdd (HAdd.hAdd b c) c')
            H : Ne (HAdd.hAdd (HMul.hMul x c) (HAdd.hAdd (HMul.hMul a c') (HMul.hMul x c') …
            ⊢ Not (Eq (HAdd.hAdd (HMul.hMul x c) (HAdd.hAdd (HMul.hMul a c') (HAdd.hAdd (H …
          -/
          simpa only [two_zsmul, add_zero, zero_add] using H
          /-
            🎉 no goals
          -/
      /-
        case a.h₂.intro.intro.intro.intro
        a b c x : Nimber
        hx : LT.lt x a
        y : Nimber
        hy : LT.lt y c
        hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
        ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
      -/
    · obtain h | h | h := lt_trichotomy (b + y) (b + c)
        /-
          case a.h₂.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y c
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd b y) (HAdd.hAdd b c)
          ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
        -/
      · have H := mul_ne_of_lt _ hx _ h
        /-
          case a.h₂.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y c
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd b y) (HAdd.hAdd b c)
          H : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (HAdd.hAdd b c)) (HMul.hMul a (HAdd. …
          ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
        -/
        rw [Nimber.mul_add x, Nimber.mul_add a, Nimber.mul_add x] at H
        /-
          case a.h₂.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y c
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd b y) (HAdd.hAdd b c)
          H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul x c)) (HAdd …
          ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
        -/
        abel_nf at H ⊢
        /-
          case a.h₂.intro.intro.intro.intro.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y c
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd b y) (HAdd.hAdd b c)
          H : Ne (HAdd.hAdd (HSMul.hSMul 2 (HMul.hMul x b)) (HAdd.hAdd (HMul.hMul x c) ( …
          ⊢ Not (Eq (HAdd.hAdd (HMul.hMul x c) (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HM …
        -/
        simpa only [two_zsmul, zero_add] using H
        /-
          🎉 no goals
        -/
        /-
          case a.h₂.intro.intro.intro.intro.inr.inl
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y c
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
          h : Eq (HAdd.hAdd b y) (HAdd.hAdd b c)
          ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
        -/
      · exact (hy.ne <| add_right_injective _ h).elim
        /-
          🎉 no goals
        -/
        /-
          case a.h₂.intro.intro.intro.intro.inr.inr
          a b c x : Nimber
          hx : LT.lt x a
          y : Nimber
          hy : LT.lt y c
          hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
          h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd b y)
          ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
        -/
      · obtain ⟨b', hb, hb'⟩ | ⟨z, hz, hz'⟩ := exists_of_lt_add h
          /-
            case a.h₂.intro.intro.intro.intro.inr.inr.inl.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y c
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd b y)
            b' : Nimber
            hb : LT.lt b' b
            hb' : Eq (HAdd.hAdd b' y) (HAdd.hAdd b c)
            ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
          -/
        · have H := mul_ne_of_lt _ hx _ hb
          /-
            case a.h₂.intro.intro.intro.intro.inr.inr.inl.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y c
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd b y)
            b' : Nimber
            hb : LT.lt b' b
            hb' : Eq (HAdd.hAdd b' y) (HAdd.hAdd b c)
            H : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a b')) (HMul.hMul x b' …
            ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
          -/
          have hb'' := add_eq_iff_eq_add.2 (add_comm b c ▸ hb')
          rw [← hb', Nimber.mul_add a b', ← add_ne_add_left (a * y), ← add_ne_add_left (a * b),
            ← add_ne_add_left (a * b'), ← hb'', Nimber.mul_add x, Nimber.mul_add x]
          /-
            case a.h₂.intro.intro.intro.intro.inr.inr.inl.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y c
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd b y)
            b' : Nimber
            hb : LT.lt b' b
            hb' : Eq (HAdd.hAdd b' y) (HAdd.hAdd b c)
            H : Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x b) (HMul.hMul a b')) (HMul.hMul x b' …
            hb'' : Eq (HAdd.hAdd (HAdd.hAdd b' y) b) c
            ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (H …
          -/
          abel_nf at H ⊢
          /-
            case a.h₂.intro.intro.intro.intro.inr.inr.inl.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y c
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd b y)
            b' : Nimber
            hb : LT.lt b' b
            hb' : Eq (HAdd.hAdd b' y) (HAdd.hAdd b c)
            hb'' : Eq (HAdd.hAdd (HAdd.hAdd b' y) b) c
            H : Ne (HAdd.hAdd (HMul.hMul x b) (HAdd.hAdd (HMul.hMul a b') (HMul.hMul x b') …
            ⊢ Not (Eq (HAdd.hAdd (HMul.hMul x b) (HAdd.hAdd (HMul.hMul a b') (HAdd.hAdd (H …
          -/
          simpa only [two_zsmul, add_zero, zero_add] using H
          /-
            🎉 no goals
          -/
          /-
            case a.h₂.intro.intro.intro.intro.inr.inr.inr.intro.intro
            a b c x : Nimber
            hx : LT.lt x a
            y : Nimber
            hy : LT.lt y c
            hx' : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul a y)) (HMul.hMul  …
            h : LT.lt (HAdd.hAdd b c) (HAdd.hAdd b y)
            z : Nimber
            hz : LT.lt z y
            hz' : Eq (HAdd.hAdd b z) (HAdd.hAdd b c)
            ⊢ Ne (HAdd.hAdd (HMul.hMul a b) (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hM …
          -/
        · exact ((hz.trans hy).ne <| add_right_injective _ hz').elim
          /-
            🎉 no goals
          -/
termination_by (a, b, c)


protected theorem add_mul (a b c : Nimber) : (a + b) * c = a * c + b * c := by
  /-
    a b c : Nimber
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
  -/
  rw [Nimber.mul_comm, Nimber.mul_add, Nimber.mul_comm, Nimber.mul_comm b]
  /-
    🎉 no goals
  -/


private theorem add_ne_zero_of_lt {a b : Nimber} (h : b < a) : a + b ≠ 0 := by
  /-
    a b : Nimber
    h : LT.lt b a
    ⊢ Ne (HAdd.hAdd a b) 0
  -/
  rw [add_ne_zero_iff]
  /-
    a b : Nimber
    h : LT.lt b a
    ⊢ Ne a b
  -/
  exact h.ne'
  /-
    🎉 no goals
  -/


protected theorem mul_assoc (a b c : Nimber) : a * b * c = a * (b * c) := by
  /-
    a b c : Nimber
    ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
  -/
  apply le_antisymm <;> refine mul_le_of_forall_ne fun x hx y hy ↦ ?_
    /-
      case a
      a b c x : Nimber
      hx : LT.lt x (HMul.hMul a b)
      y : Nimber
      hy : LT.lt y c
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x c) (HMul.hMul (HMul.hMul a b) y)) (HMu …
    -/
  · obtain ⟨a', ha, b', hb, rfl⟩ := exists_of_lt_mul hx
    have H : (a + a') * ((b + b') * (c + y)) ≠ 0 := by
      apply mul_ne_zero _ (mul_ne_zero _ _) <;> apply add_ne_zero_of_lt
      assumption'
    /-
      case a.intro.intro.intro.intro
      a b c y : Nimber
      hy : LT.lt y c
      a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      hb : LT.lt b' b
      hx : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a b')) (HMul.hMul …
      H : Ne (HMul.hMul (HAdd.hAdd a a') (HMul.hMul (HAdd.hAdd b b') (HAdd.hAdd c y) …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) ( …
    -/
    simp only [Nimber.add_mul, Nimber.mul_add] at H ⊢
    /-
      case a.intro.intro.intro.intro
      a b c y : Nimber
      hy : LT.lt y c
      a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      hb : LT.lt b' b
      hx : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a b')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul b c)) (HMul.hM …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul a' b) c …
    -/
    iterate 7 rw [Nimber.mul_assoc]
    /-
      case a.intro.intro.intro.intro
      a b c y : Nimber
      hy : LT.lt y c
      a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      hb : LT.lt b' b
      hx : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a b')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul b c)) (HMul.hM …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' (HMul.hMul b c) …
    -/
    rw [← add_ne_add_left (a * (b * c))]
    /-
      case a.intro.intro.intro.intro
      a b c y : Nimber
      hy : LT.lt y c
      a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      hb : LT.lt b' b
      hx : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a b')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HMul.hMul b c)) (HMul.hM …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' (HMu …
    -/
    abel_nf at H ⊢
    /-
      case a.intro.intro.intro.intro
      a b c y : Nimber
      hy : LT.lt y c
      a' : Nimber
      ha : LT.lt a' a
      b' : Nimber
      hb : LT.lt b' b
      hx : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul a' b) (HMul.hMul a b')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HMul.hMul a (HMul.hMul b c)) (HAdd.hAdd (HMul.hMul a' (HMul …
      ⊢ Not (Eq (HAdd.hAdd (HMul.hMul a (HMul.hMul b c)) (HAdd.hAdd (HMul.hMul a' (H …
    -/
    simpa only [two_zsmul, zero_add] using H
    /-
      🎉 no goals
    -/
    /-
      case a
      a b c x : Nimber
      hx : LT.lt x a
      y : Nimber
      hy : LT.lt y (HMul.hMul b c)
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (HMul.hMul b c)) (HMul.hMul a y)) (HMu …
    -/
  · obtain ⟨b', hb, c', hc, rfl⟩ := exists_of_lt_mul hy
    have H : (a + x) * (b + b') * (c + c') ≠ 0 := by
      apply mul_ne_zero (mul_ne_zero _ _) <;> apply add_ne_zero_of_lt
      assumption'
    /-
      case a.intro.intro.intro.intro
      a b c x : Nimber
      hx : LT.lt x a
      b' : Nimber
      hb : LT.lt b' b
      c' : Nimber
      hc : LT.lt c' c
      hy : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul b' c) (HMul.hMul b c')) (HMul.hMul …
      H : Ne (HMul.hMul (HMul.hMul (HAdd.hAdd a x) (HAdd.hAdd b b')) (HAdd.hAdd c c' …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (HMul.hMul b c)) (HMul.hMul a (HAdd.hA …
    -/
    simp only [Nimber.add_mul, Nimber.mul_add] at H ⊢
    /-
      case a.intro.intro.intro.intro
      a b c x : Nimber
      hx : LT.lt x a
      b' : Nimber
      hb : LT.lt b' b
      c' : Nimber
      hc : LT.lt c' c
      hy : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul b' c) (HMul.hMul b c')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul a b) c) (HMul.hM …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x (HMul.hMul b c)) (HAdd.hAdd (HAdd.hAdd …
    -/
    iterate 7 rw [← Nimber.mul_assoc]
    /-
      case a.intro.intro.intro.intro
      a b c x : Nimber
      hx : LT.lt x a
      b' : Nimber
      hb : LT.lt b' b
      c' : Nimber
      hc : LT.lt c' c
      hy : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul b' c) (HMul.hMul b c')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul a b) c) (HMul.hM …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul x b) c) (HAdd.hAdd (HAdd.hAdd …
    -/
    rw [← add_ne_add_left (a * b * c)]
    /-
      case a.intro.intro.intro.intro
      a b c x : Nimber
      hx : LT.lt x a
      b' : Nimber
      hb : LT.lt b' b
      c' : Nimber
      hc : LT.lt c' c
      hy : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul b' c) (HMul.hMul b c')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul a b) c) (HMul.hM …
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul x b) c) (HAdd.hAdd …
    -/
    abel_nf at H ⊢
    /-
      case a.intro.intro.intro.intro
      a b c x : Nimber
      hx : LT.lt x a
      b' : Nimber
      hb : LT.lt b' b
      c' : Nimber
      hc : LT.lt c' c
      hy : LT.lt (HAdd.hAdd (HAdd.hAdd (HMul.hMul b' c) (HMul.hMul b c')) (HMul.hMul …
      H : Ne (HAdd.hAdd (HMul.hMul (HMul.hMul a b) c) (HAdd.hAdd (HMul.hMul (HMul.hM …
      ⊢ Not (Eq (HAdd.hAdd (HMul.hMul (HMul.hMul a b) c) (HAdd.hAdd (HMul.hMul (HMul …
    -/
    simpa only [two_zsmul, zero_add] using H
    /-
      🎉 no goals
    -/
termination_by (a, b, c)


instance : IsCancelMulZero Nimber where
  mul_left_cancel_of_ne_zero ha h := by
    /-
      a b c : Nimber
      a✝ b✝ c✝ : Nimber
      ha : Ne a✝ 0
      h : Eq (HMul.hMul a✝ b✝) (HMul.hMul a✝ c✝)
      ⊢ Eq b✝ c✝
    -/
    rw [← add_eq_zero, ← Nimber.mul_add, mul_eq_zero] at h
    /-
      a b c : Nimber
      a✝ b✝ c✝ : Nimber
      ha : Ne a✝ 0
      h : Or (Eq a✝ 0) (Eq (HAdd.hAdd b✝ c✝) 0)
      ⊢ Eq b✝ c✝
    -/
    exact add_eq_zero.1 (h.resolve_left ha)
    /-
      🎉 no goals
    -/
  mul_right_cancel_of_ne_zero ha h := by
    /-
      a b c : Nimber
      a✝ b✝ c✝ : Nimber
      ha : Ne b✝ 0
      h : Eq (HMul.hMul a✝ b✝) (HMul.hMul c✝ b✝)
      ⊢ Eq a✝ c✝
    -/
    rw [← add_eq_zero, ← Nimber.add_mul, mul_eq_zero] at h
    /-
      a b c : Nimber
      a✝ b✝ c✝ : Nimber
      ha : Ne b✝ 0
      h : Or (Eq (HAdd.hAdd a✝ c✝) 0) (Eq b✝ 0)
      ⊢ Eq a✝ c✝
    -/
    exact add_eq_zero.1 (h.resolve_right ha)
    /-
      🎉 no goals
    -/


protected theorem one_mul (a : Nimber) : 1 * a = a := by
  /-
    a : Nimber
    ⊢ Eq (HMul.hMul 1 a) a
  -/
  apply le_antisymm
    /-
      case a
      a : Nimber
      ⊢ LE.le (HMul.hMul 1 a) a
    -/
  · refine mul_le_of_forall_ne fun x hx y hy ↦ ?_
    /-
      case a
      a x : Nimber
      hx : LT.lt x 1
      y : Nimber
      hy : LT.lt y a
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x a) (HMul.hMul 1 y)) (HMul.hMul x y)) a
    -/
    rw [Nimber.lt_one_iff_zero] at hx
    /-
      case a
      a x : Nimber
      hx : Eq x 0
      y : Nimber
      hy : LT.lt y a
      ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HMul.hMul x a) (HMul.hMul 1 y)) (HMul.hMul x y)) a
    -/
    rw [hx, Nimber.one_mul, zero_mul, zero_mul, add_zero, zero_add]
    /-
      case a
      a x : Nimber
      hx : Eq x 0
      y : Nimber
      hy : LT.lt y a
      ⊢ Ne y a
    -/
    exact hy.ne
    /-
      🎉 no goals
    -/
    /-
      case a
      a : Nimber
      ⊢ LE.le a (HMul.hMul 1 a)
    -/
  · by_contra! h
    /-
      case a
      a : Nimber
      h : LT.lt (HMul.hMul 1 a) a
      ⊢ False
    -/
    replace h := h -- needed to remind `termination_by`
    /-
      case a
      a : Nimber
      h : LT.lt (HMul.hMul 1 a) a
      ⊢ False
    -/
    exact (mul_left_cancel₀ one_ne_zero <| Nimber.one_mul _).not_lt h
    /-
      🎉 no goals
    -/
termination_by a


protected theorem mul_one (a : Nimber) : a * 1 = a := by
  /-
    a : Nimber
    ⊢ Eq (HMul.hMul a 1) a
  -/
  rw [Nimber.mul_comm, Nimber.one_mul]
  /-
    🎉 no goals
  -/


instance : CommRing Nimber where
  left_distrib := Nimber.mul_add
  right_distrib := Nimber.add_mul
  zero_mul := zero_mul
  mul_zero := mul_zero
  mul_assoc := Nimber.mul_assoc
  mul_comm := Nimber.mul_comm
  one_mul := Nimber.one_mul
  mul_one := Nimber.mul_one
  __ : AddCommGroupWithOne Nimber := inferInstance


instance : IsDomain Nimber where

instance : CancelMonoidWithZero Nimber where


