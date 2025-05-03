/-- A typeclass saying that `(p : R × R) ↦ p.1 + p.2` maps any product of bounded sets to a bounded
set. This property follows from `LipschitzAdd`, and thus automatically holds, e.g., for seminormed
additive groups. -/
class BoundedAdd (R : Type*) [Bornology R] [Add R] : Prop where
  isBounded_add : ∀ {s t : Set R},
    Bornology.IsBounded s → Bornology.IsBounded t → Bornology.IsBounded (s + t)


lemma isBounded_add [Bornology R] [Add R] [BoundedAdd R] {s t : Set R}
    (hs : Bornology.IsBounded s) (ht : Bornology.IsBounded t) :
    Bornology.IsBounded (s + t) := BoundedAdd.isBounded_add hs ht


lemma add_bounded_of_bounded_of_bounded {X : Type*} [PseudoMetricSpace R] [Add R] [BoundedAdd R]
    {f g : X → R} (f_bdd : ∃ C, ∀ x y, dist (f x) (f y) ≤ C)
    (g_bdd : ∃ C, ∀ x y, dist (g x) (g y) ≤ C) :
    ∃ C, ∀ x y, dist ((f + g) x) ((f + g) y) ≤ C := by
  obtain ⟨C, hC⟩ := Metric.isBounded_iff.mp <|
    isBounded_add (Metric.isBounded_range_iff.mpr f_bdd) (Metric.isBounded_range_iff.mpr g_bdd)
  /-
    case intro
    R : Type u_1
    X : Type u_2
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Add R
    inst✝ : BoundedAdd R
    f g : X → R
    f_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (f x) (f y)) C
    g_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (g x) (g y)) C
    C : Real
    hC : ∀ ⦃x : R⦄, Membership.mem (HAdd.hAdd (Set.range f) (Set.range g)) x → ∀ ⦃ …
    ⊢ Exists fun C => ∀ (x y : X), LE.le (Dist.dist (HAdd.hAdd f g x) (HAdd.hAdd f …
  -/
  use C
  /-
    case h
    R : Type u_1
    X : Type u_2
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Add R
    inst✝ : BoundedAdd R
    f g : X → R
    f_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (f x) (f y)) C
    g_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (g x) (g y)) C
    C : Real
    hC : ∀ ⦃x : R⦄, Membership.mem (HAdd.hAdd (Set.range f) (Set.range g)) x → ∀ ⦃ …
    ⊢ ∀ (x y : X), LE.le (Dist.dist (HAdd.hAdd f g x) (HAdd.hAdd f g y)) C
  -/
  intro x y
  exact hC (Set.add_mem_add (Set.mem_range_self (f := f) x) (Set.mem_range_self (f := g) x))
           (Set.add_mem_add (Set.mem_range_self (f := f) y) (Set.mem_range_self (f := g) y))


instance [PseudoMetricSpace R] [AddMonoid R] [LipschitzAdd R] : BoundedAdd R where
  isBounded_add {s t} s_bdd t_bdd := by
    /-
      R : Type u_1
      inst✝² : PseudoMetricSpace R
      inst✝¹ : AddMonoid R
      inst✝ : LipschitzAdd R
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      ⊢ Bornology.IsBounded (HAdd.hAdd s t)
    -/
    have bdd : Bornology.IsBounded (s ×ˢ t) := Bornology.IsBounded.prod s_bdd t_bdd
    /-
      R : Type u_1
      inst✝² : PseudoMetricSpace R
      inst✝¹ : AddMonoid R
      inst✝ : LipschitzAdd R
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      ⊢ Bornology.IsBounded (HAdd.hAdd s t)
    -/
    obtain ⟨C, add_lip⟩ := ‹LipschitzAdd R›.lipschitz_add
    /-
      case intro
      R : Type u_1
      inst✝² : PseudoMetricSpace R
      inst✝¹ : AddMonoid R
      inst✝ : LipschitzAdd R
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      C : NNReal
      add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
      ⊢ Bornology.IsBounded (HAdd.hAdd s t)
    -/
    convert add_lip.isBounded_image bdd
    /-
      case h.e'_3
      R : Type u_1
      inst✝² : PseudoMetricSpace R
      inst✝¹ : AddMonoid R
      inst✝ : LipschitzAdd R
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      C : NNReal
      add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
      ⊢ Eq (HAdd.hAdd s t) (Set.image (fun p => HAdd.hAdd p.1 p.2) (SProd.sprod s t))
    -/
    ext p
    /-
      case h.e'_3.h
      R : Type u_1
      inst✝² : PseudoMetricSpace R
      inst✝¹ : AddMonoid R
      inst✝ : LipschitzAdd R
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      C : NNReal
      add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
      p : R
      ⊢ Iff (Membership.mem (HAdd.hAdd s t) p) (Membership.mem (Set.image (fun p =>  …
    -/
    simp only [Set.mem_image, Set.mem_prod, Prod.exists]
    /-
      case h.e'_3.h
      R : Type u_1
      inst✝² : PseudoMetricSpace R
      inst✝¹ : AddMonoid R
      inst✝ : LipschitzAdd R
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      C : NNReal
      add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
      p : R
      ⊢ Iff (Membership.mem (HAdd.hAdd s t) p) (Exists fun a => Exists fun b => And  …
    -/
    constructor
      /-
        case h.e'_3.h.mp
        R : Type u_1
        inst✝² : PseudoMetricSpace R
        inst✝¹ : AddMonoid R
        inst✝ : LipschitzAdd R
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        C : NNReal
        add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
        p : R
        ⊢ Membership.mem (HAdd.hAdd s t) p → Exists fun a => Exists fun b => And (And  …
      -/
    · intro ⟨a, a_in_s, b, b_in_t, eq_p⟩
      /-
        case h.e'_3.h.mp
        R : Type u_1
        inst✝² : PseudoMetricSpace R
        inst✝¹ : AddMonoid R
        inst✝ : LipschitzAdd R
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        C : NNReal
        add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
        p a : R
        a_in_s : Membership.mem s a
        b : R
        b_in_t : Membership.mem t b
        eq_p : Eq ((fun x1 x2 => HAdd.hAdd x1 x2) a b) p
        ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.me …
      -/
      exact ⟨a, b, ⟨a_in_s, b_in_t⟩, eq_p⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.mpr
        R : Type u_1
        inst✝² : PseudoMetricSpace R
        inst✝¹ : AddMonoid R
        inst✝ : LipschitzAdd R
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        C : NNReal
        add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
        p : R
        ⊢ (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.m …
      -/
    · intro ⟨a, b, ⟨a_in_s, b_in_t⟩, eq_p⟩
      /-
        case h.e'_3.h.mpr
        R : Type u_1
        inst✝² : PseudoMetricSpace R
        inst✝¹ : AddMonoid R
        inst✝ : LipschitzAdd R
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        C : NNReal
        add_lip : LipschitzWith C fun p => HAdd.hAdd p.1 p.2
        p a b : R
        a_in_s : Membership.mem s a
        b_in_t : Membership.mem t b
        eq_p : Eq (HAdd.hAdd a b) p
        ⊢ Membership.mem (HAdd.hAdd s t) p
      -/
      simpa [← eq_p] using Set.add_mem_add a_in_s b_in_t
      /-
        🎉 no goals
      -/


/-- A typeclass saying that `(p : R × R) ↦ p.1 - p.2` maps any product of bounded sets to a bounded
set. This property automatically holds for seminormed additive groups, but it also holds, e.g.,
for `ℝ≥0`. -/
class BoundedSub (R : Type*) [Bornology R] [Sub R] : Prop where
  isBounded_sub : ∀ {s t : Set R},
    Bornology.IsBounded s → Bornology.IsBounded t → Bornology.IsBounded (s - t)


lemma isBounded_sub [Bornology R] [Sub R] [BoundedSub R] {s t : Set R}
    (hs : Bornology.IsBounded s) (ht : Bornology.IsBounded t) :
    Bornology.IsBounded (s - t) := BoundedSub.isBounded_sub hs ht


lemma sub_bounded_of_bounded_of_bounded {X : Type*} [PseudoMetricSpace R] [Sub R] [BoundedSub R]
    {f g : X → R} (f_bdd : ∃ C, ∀ x y, dist (f x) (f y) ≤ C)
    (g_bdd : ∃ C, ∀ x y, dist (g x) (g y) ≤ C) :
    ∃ C, ∀ x y, dist ((f - g) x) ((f - g) y) ≤ C := by
  obtain ⟨C, hC⟩ := Metric.isBounded_iff.mp <|
    isBounded_sub (Metric.isBounded_range_iff.mpr f_bdd) (Metric.isBounded_range_iff.mpr g_bdd)
  /-
    case intro
    R : Type u_1
    X : Type u_2
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Sub R
    inst✝ : BoundedSub R
    f g : X → R
    f_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (f x) (f y)) C
    g_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (g x) (g y)) C
    C : Real
    hC : ∀ ⦃x : R⦄, Membership.mem (HSub.hSub (Set.range f) (Set.range g)) x → ∀ ⦃ …
    ⊢ Exists fun C => ∀ (x y : X), LE.le (Dist.dist (HSub.hSub f g x) (HSub.hSub f …
  -/
  use C
  /-
    case h
    R : Type u_1
    X : Type u_2
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Sub R
    inst✝ : BoundedSub R
    f g : X → R
    f_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (f x) (f y)) C
    g_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (g x) (g y)) C
    C : Real
    hC : ∀ ⦃x : R⦄, Membership.mem (HSub.hSub (Set.range f) (Set.range g)) x → ∀ ⦃ …
    ⊢ ∀ (x y : X), LE.le (Dist.dist (HSub.hSub f g x) (HSub.hSub f g y)) C
  -/
  intro x y
  exact hC (Set.sub_mem_sub (Set.mem_range_self (f := f) x) (Set.mem_range_self (f := g) x))
           (Set.sub_mem_sub (Set.mem_range_self (f := f) y) (Set.mem_range_self (f := g) y))


lemma boundedSub_of_lipschitzWith_sub [PseudoMetricSpace R] [Sub R] {K : NNReal}
    (lip : LipschitzWith K (fun (p : R × R) ↦ p.1 - p.2)) :
    BoundedSub R where
  isBounded_sub {s t} s_bdd t_bdd := by
    /-
      R : Type u_1
      inst✝¹ : PseudoMetricSpace R
      inst✝ : Sub R
      K : NNReal
      lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      ⊢ Bornology.IsBounded (HSub.hSub s t)
    -/
    have bdd : Bornology.IsBounded (s ×ˢ t) := Bornology.IsBounded.prod s_bdd t_bdd
    /-
      R : Type u_1
      inst✝¹ : PseudoMetricSpace R
      inst✝ : Sub R
      K : NNReal
      lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      ⊢ Bornology.IsBounded (HSub.hSub s t)
    -/
    convert lip.isBounded_image bdd
    /-
      case h.e'_3
      R : Type u_1
      inst✝¹ : PseudoMetricSpace R
      inst✝ : Sub R
      K : NNReal
      lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      ⊢ Eq (HSub.hSub s t) (Set.image (fun p => HSub.hSub p.1 p.2) (SProd.sprod s t))
    -/
    ext p
    /-
      case h.e'_3.h
      R : Type u_1
      inst✝¹ : PseudoMetricSpace R
      inst✝ : Sub R
      K : NNReal
      lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      p : R
      ⊢ Iff (Membership.mem (HSub.hSub s t) p) (Membership.mem (Set.image (fun p =>  …
    -/
    simp only [Set.mem_image, Set.mem_prod, Prod.exists]
    /-
      case h.e'_3.h
      R : Type u_1
      inst✝¹ : PseudoMetricSpace R
      inst✝ : Sub R
      K : NNReal
      lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
      s t : Set R
      s_bdd : Bornology.IsBounded s
      t_bdd : Bornology.IsBounded t
      bdd : Bornology.IsBounded (SProd.sprod s t)
      p : R
      ⊢ Iff (Membership.mem (HSub.hSub s t) p) (Exists fun a => Exists fun b => And  …
    -/
    constructor
      /-
        case h.e'_3.h.mp
        R : Type u_1
        inst✝¹ : PseudoMetricSpace R
        inst✝ : Sub R
        K : NNReal
        lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        p : R
        ⊢ Membership.mem (HSub.hSub s t) p → Exists fun a => Exists fun b => And (And  …
      -/
    · intro ⟨a, a_in_s, b, b_in_t, eq_p⟩
      /-
        case h.e'_3.h.mp
        R : Type u_1
        inst✝¹ : PseudoMetricSpace R
        inst✝ : Sub R
        K : NNReal
        lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        p a : R
        a_in_s : Membership.mem s a
        b : R
        b_in_t : Membership.mem t b
        eq_p : Eq ((fun x1 x2 => HSub.hSub x1 x2) a b) p
        ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.me …
      -/
      exact ⟨a, b, ⟨a_in_s, b_in_t⟩, eq_p⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3.h.mpr
        R : Type u_1
        inst✝¹ : PseudoMetricSpace R
        inst✝ : Sub R
        K : NNReal
        lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        p : R
        ⊢ (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membership.m …
      -/
    · intro ⟨a, b, ⟨a_in_s, b_in_t⟩, eq_p⟩
      /-
        case h.e'_3.h.mpr
        R : Type u_1
        inst✝¹ : PseudoMetricSpace R
        inst✝ : Sub R
        K : NNReal
        lip : LipschitzWith K fun p => HSub.hSub p.1 p.2
        s t : Set R
        s_bdd : Bornology.IsBounded s
        t_bdd : Bornology.IsBounded t
        bdd : Bornology.IsBounded (SProd.sprod s t)
        p a b : R
        a_in_s : Membership.mem s a
        b_in_t : Membership.mem t b
        eq_p : Eq (HSub.hSub a b) p
        ⊢ Membership.mem (HSub.hSub s t) p
      -/
      simpa [← eq_p] using Set.sub_mem_sub a_in_s b_in_t
      /-
        🎉 no goals
      -/


/-- A typeclass saying that `(p : R × R) ↦ p.1 * p.2` maps any product of bounded sets to a bounded
set. This property automatically holds for non-unital seminormed rings, but it also holds, e.g.,
for `ℝ≥0`. -/
class BoundedMul (R : Type*) [Bornology R] [Mul R] : Prop where
  isBounded_mul : ∀ {s t : Set R},
    Bornology.IsBounded s → Bornology.IsBounded t → Bornology.IsBounded (s * t)


lemma isBounded_mul [Bornology R] [Mul R] [BoundedMul R] {s t : Set R}
    (hs : Bornology.IsBounded s) (ht : Bornology.IsBounded t) :
    Bornology.IsBounded (s * t) := BoundedMul.isBounded_mul hs ht


lemma isBounded_pow {R : Type*} [Bornology R] [Monoid R] [BoundedMul R] {s : Set R}
    (s_bdd : Bornology.IsBounded s) (n : ℕ) :
    Bornology.IsBounded ((fun x ↦ x ^ n) '' s) := by
  /-
    R : Type u_2
    inst✝² : Bornology R
    inst✝¹ : Monoid R
    inst✝ : BoundedMul R
    s : Set R
    s_bdd : Bornology.IsBounded s
    n : Nat
    ⊢ Bornology.IsBounded (Set.image (fun x => HPow.hPow x n) s)
  -/
  induction' n with n hn
    /-
      case zero
      R : Type u_2
      inst✝² : Bornology R
      inst✝¹ : Monoid R
      inst✝ : BoundedMul R
      s : Set R
      s_bdd : Bornology.IsBounded s
      ⊢ Bornology.IsBounded (Set.image (fun x => HPow.hPow x 0) s)
    -/
  · by_cases s_empty : s = ∅
      /-
        case pos
        R : Type u_2
        inst✝² : Bornology R
        inst✝¹ : Monoid R
        inst✝ : BoundedMul R
        s : Set R
        s_bdd : Bornology.IsBounded s
        s_empty : Eq s EmptyCollection.emptyCollection
        ⊢ Bornology.IsBounded (Set.image (fun x => HPow.hPow x 0) s)
      -/
    · simp [s_empty]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_2
      inst✝² : Bornology R
      inst✝¹ : Monoid R
      inst✝ : BoundedMul R
      s : Set R
      s_bdd : Bornology.IsBounded s
      s_empty : Not (Eq s EmptyCollection.emptyCollection)
      ⊢ Bornology.IsBounded (Set.image (fun x => HPow.hPow x 0) s)
    -/
    simp_rw [← nonempty_iff_ne_empty] at s_empty
    /-
      case neg
      R : Type u_2
      inst✝² : Bornology R
      inst✝¹ : Monoid R
      inst✝ : BoundedMul R
      s : Set R
      s_bdd : Bornology.IsBounded s
      s_empty : s.Nonempty
      ⊢ Bornology.IsBounded (Set.image (fun x => HPow.hPow x 0) s)
    -/
    simp [s_empty]
    /-
      🎉 no goals
    -/
  · have obs : ((fun x ↦ x ^ (n + 1)) '' s) ⊆ ((fun x ↦ x ^ n) '' s) * s := by
      intro x hx
      simp only [mem_image] at hx
      obtain ⟨y, y_in_s, ypow_eq_x⟩ := hx
      rw [← ypow_eq_x, pow_succ y n]
      apply Set.mul_mem_mul _ y_in_s
      use y
    /-
      case succ
      R : Type u_2
      inst✝² : Bornology R
      inst✝¹ : Monoid R
      inst✝ : BoundedMul R
      s : Set R
      s_bdd : Bornology.IsBounded s
      n : Nat
      hn : Bornology.IsBounded (Set.image (fun x => HPow.hPow x n) s)
      obs : HasSubset.Subset (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) s) (H …
      ⊢ Bornology.IsBounded (Set.image (fun x => HPow.hPow x (HAdd.hAdd n 1)) s)
    -/
    exact (isBounded_mul hn s_bdd).subset obs
    /-
      🎉 no goals
    -/


lemma mul_bounded_of_bounded_of_bounded {X : Type*} [PseudoMetricSpace R] [Mul R] [BoundedMul R]
    {f g : X → R} (f_bdd : ∃ C, ∀ x y, dist (f x) (f y) ≤ C)
    (g_bdd : ∃ C, ∀ x y, dist (g x) (g y) ≤ C) :
    ∃ C, ∀ x y, dist ((f * g) x) ((f * g) y) ≤ C := by
  obtain ⟨C, hC⟩ := Metric.isBounded_iff.mp <|
    isBounded_mul (Metric.isBounded_range_iff.mpr f_bdd) (Metric.isBounded_range_iff.mpr g_bdd)
  /-
    case intro
    R : Type u_1
    X : Type u_2
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Mul R
    inst✝ : BoundedMul R
    f g : X → R
    f_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (f x) (f y)) C
    g_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (g x) (g y)) C
    C : Real
    hC : ∀ ⦃x : R⦄, Membership.mem (HMul.hMul (Set.range f) (Set.range g)) x → ∀ ⦃ …
    ⊢ Exists fun C => ∀ (x y : X), LE.le (Dist.dist (HMul.hMul f g x) (HMul.hMul f …
  -/
  use C
  /-
    case h
    R : Type u_1
    X : Type u_2
    inst✝² : PseudoMetricSpace R
    inst✝¹ : Mul R
    inst✝ : BoundedMul R
    f g : X → R
    f_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (f x) (f y)) C
    g_bdd : Exists fun C => ∀ (x y : X), LE.le (Dist.dist (g x) (g y)) C
    C : Real
    hC : ∀ ⦃x : R⦄, Membership.mem (HMul.hMul (Set.range f) (Set.range g)) x → ∀ ⦃ …
    ⊢ ∀ (x y : X), LE.le (Dist.dist (HMul.hMul f g x) (HMul.hMul f g y)) C
  -/
  intro x y
  exact hC (Set.mul_mem_mul (Set.mem_range_self (f := f) x) (Set.mem_range_self (f := g) x))
           (Set.mul_mem_mul (Set.mem_range_self (f := f) y) (Set.mem_range_self (f := g) y))


lemma SeminormedAddCommGroup.lipschitzWith_sub :
    LipschitzWith 2 (fun (p : R × R) ↦ p.1 - p.2) := by
  /-
    R : Type u_1
    inst✝ : SeminormedAddCommGroup R
    ⊢ LipschitzWith 2 fun p => HSub.hSub p.1 p.2
  -/
  convert LipschitzWith.prod_fst.sub LipschitzWith.prod_snd
  /-
    case h.e'_5
    R : Type u_1
    inst✝ : SeminormedAddCommGroup R
    ⊢ Eq 2 (HAdd.hAdd 1 1)
  -/
  norm_num
  /-
    🎉 no goals
  -/


instance : BoundedSub R := boundedSub_of_lipschitzWith_sub SeminormedAddCommGroup.lipschitzWith_sub


instance : BoundedMul R where
  isBounded_mul {s t} hs ht := by
    /-
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      ⊢ Bornology.IsBounded (HMul.hMul s t)
    -/
    obtain ⟨Af, hAf⟩ := (Metric.isBounded_iff_subset_closedBall 0).mp hs
    /-
      case intro
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      ⊢ Bornology.IsBounded (HMul.hMul s t)
    -/
    obtain ⟨Ag, hAg⟩ := (Metric.isBounded_iff_subset_closedBall 0).mp ht
    /-
      case intro.intro
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      ⊢ Bornology.IsBounded (HMul.hMul s t)
    -/
    rw [Metric.isBounded_iff] at hs ht ⊢
    /-
      case intro.intro
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Exists fun C => ∀ ⦃x : R⦄, Membership.mem s x → ∀ ⦃y : R⦄, Membership.mem …
      ht : Exists fun C => ∀ ⦃x : R⦄, Membership.mem t x → ∀ ⦃y : R⦄, Membership.mem …
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      ⊢ Exists fun C => ∀ ⦃x : R⦄, Membership.mem (HMul.hMul s t) x → ∀ ⦃y : R⦄, Mem …
    -/
    use 2 * Af * Ag
    /-
      case h
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Exists fun C => ∀ ⦃x : R⦄, Membership.mem s x → ∀ ⦃y : R⦄, Membership.mem …
      ht : Exists fun C => ∀ ⦃x : R⦄, Membership.mem t x → ∀ ⦃y : R⦄, Membership.mem …
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      ⊢ ∀ ⦃x : R⦄, Membership.mem (HMul.hMul s t) x → ∀ ⦃y : R⦄, Membership.mem (HMu …
    -/
    intro z hz w hw
    /-
      case h
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Exists fun C => ∀ ⦃x : R⦄, Membership.mem s x → ∀ ⦃y : R⦄, Membership.mem …
      ht : Exists fun C => ∀ ⦃x : R⦄, Membership.mem t x → ∀ ⦃y : R⦄, Membership.mem …
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      z : R
      hz : Membership.mem (HMul.hMul s t) z
      w : R
      hw : Membership.mem (HMul.hMul s t) w
      ⊢ LE.le (Dist.dist z w) (HMul.hMul (HMul.hMul 2 Af) Ag)
    -/
    obtain ⟨x₁, hx₁, y₁, hy₁, z_eq⟩ := Set.mem_mul.mp hz
    /-
      case h.intro.intro.intro.intro
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Exists fun C => ∀ ⦃x : R⦄, Membership.mem s x → ∀ ⦃y : R⦄, Membership.mem …
      ht : Exists fun C => ∀ ⦃x : R⦄, Membership.mem t x → ∀ ⦃y : R⦄, Membership.mem …
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      z : R
      hz : Membership.mem (HMul.hMul s t) z
      w : R
      hw : Membership.mem (HMul.hMul s t) w
      x₁ : R
      hx₁ : Membership.mem s x₁
      y₁ : R
      hy₁ : Membership.mem t y₁
      z_eq : Eq (HMul.hMul x₁ y₁) z
      ⊢ LE.le (Dist.dist z w) (HMul.hMul (HMul.hMul 2 Af) Ag)
    -/
    obtain ⟨x₂, hx₂, y₂, hy₂, w_eq⟩ := Set.mem_mul.mp hw
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Exists fun C => ∀ ⦃x : R⦄, Membership.mem s x → ∀ ⦃y : R⦄, Membership.mem …
      ht : Exists fun C => ∀ ⦃x : R⦄, Membership.mem t x → ∀ ⦃y : R⦄, Membership.mem …
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      z : R
      hz : Membership.mem (HMul.hMul s t) z
      w : R
      hw : Membership.mem (HMul.hMul s t) w
      x₁ : R
      hx₁ : Membership.mem s x₁
      y₁ : R
      hy₁ : Membership.mem t y₁
      z_eq : Eq (HMul.hMul x₁ y₁) z
      x₂ : R
      hx₂ : Membership.mem s x₂
      y₂ : R
      hy₂ : Membership.mem t y₂
      w_eq : Eq (HMul.hMul x₂ y₂) w
      ⊢ LE.le (Dist.dist z w) (HMul.hMul (HMul.hMul 2 Af) Ag)
    -/
    rw [← w_eq, ← z_eq, dist_eq_norm]
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝ : NonUnitalSeminormedRing R
      s t : Set R
      hs : Exists fun C => ∀ ⦃x : R⦄, Membership.mem s x → ∀ ⦃y : R⦄, Membership.mem …
      ht : Exists fun C => ∀ ⦃x : R⦄, Membership.mem t x → ∀ ⦃y : R⦄, Membership.mem …
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      z : R
      hz : Membership.mem (HMul.hMul s t) z
      w : R
      hw : Membership.mem (HMul.hMul s t) w
      x₁ : R
      hx₁ : Membership.mem s x₁
      y₁ : R
      hy₁ : Membership.mem t y₁
      z_eq : Eq (HMul.hMul x₁ y₁) z
      x₂ : R
      hx₂ : Membership.mem s x₂
      y₂ : R
      hy₂ : Membership.mem t y₂
      w_eq : Eq (HMul.hMul x₂ y₂) w
      ⊢ LE.le (Norm.norm (HSub.hSub (HMul.hMul x₁ y₁) (HMul.hMul x₂ y₂))) (HMul.hMul …
    -/
    have hAf' : 0 ≤ Af := Metric.nonempty_closedBall.mp ⟨_, hAf hx₁⟩
    have aux : ∀ {x y}, x ∈ s → y ∈ t → ‖x * y‖ ≤ Af * Ag := by
      intro x y x_in_s y_in_t
      apply (norm_mul_le _ _).trans (mul_le_mul _ _ (norm_nonneg _) hAf')
      · exact mem_closedBall_zero_iff.mp (hAf x_in_s)
      · exact mem_closedBall_zero_iff.mp (hAg y_in_t)
    calc ‖x₁ * y₁ - x₂ * y₂‖
     _ ≤ ‖x₁ * y₁‖ + ‖x₂ * y₂‖        := norm_sub_le _ _
     _ ≤ Af * Ag + Af * Ag            := add_le_add (aux hx₁ hy₁) (aux hx₂ hy₂)
     _ = 2 * Af * Ag                  := by simp [← two_mul, mul_assoc]


instance : BoundedSub ℝ≥0 := boundedSub_of_lipschitzWith_sub NNReal.lipschitzWith_sub


open Metric in
instance : BoundedMul ℝ≥0 where
  isBounded_mul {s t} hs ht := by
    /-
      s t : Set NNReal
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      ⊢ Bornology.IsBounded (HMul.hMul s t)
    -/
    obtain ⟨Af, hAf⟩ := (isBounded_iff_subset_closedBall 0).mp hs
    /-
      case intro
      s t : Set NNReal
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      ⊢ Bornology.IsBounded (HMul.hMul s t)
    -/
    obtain ⟨Ag, hAg⟩ := (isBounded_iff_subset_closedBall 0).mp ht
    have key : IsCompact (closedBall (0 : ℝ≥0) Af ×ˢ closedBall (0 : ℝ≥0) Ag) :=
      IsCompact.prod (isCompact_closedBall _ _) (isCompact_closedBall _ _)
    /-
      case intro.intro
      s t : Set NNReal
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      key : IsCompact (SProd.sprod (Metric.closedBall 0 Af) (Metric.closedBall 0 Ag))
      ⊢ Bornology.IsBounded (HMul.hMul s t)
    -/
    apply Bornology.IsBounded.subset (key.image continuous_mul).isBounded
    /-
      case intro.intro
      s t : Set NNReal
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      key : IsCompact (SProd.sprod (Metric.closedBall 0 Af) (Metric.closedBall 0 Ag))
      ⊢ HasSubset.Subset (HMul.hMul s t) (Set.image (fun p => HMul.hMul p.1 p.2) (SP …
    -/
    intro _ ⟨x, x_in_s, y, y_in_t, xy_eq⟩
    /-
      case intro.intro
      s t : Set NNReal
      hs : Bornology.IsBounded s
      ht : Bornology.IsBounded t
      Af : Real
      hAf : HasSubset.Subset s (Metric.closedBall 0 Af)
      Ag : Real
      hAg : HasSubset.Subset t (Metric.closedBall 0 Ag)
      key : IsCompact (SProd.sprod (Metric.closedBall 0 Af) (Metric.closedBall 0 Ag))
      a✝ x : NNReal
      x_in_s : Membership.mem s x
      y : NNReal
      y_in_t : Membership.mem t y
      xy_eq : Eq ((fun x1 x2 => HMul.hMul x1 x2) x y) a✝
      ⊢ Membership.mem (Set.image (fun p => HMul.hMul p.1 p.2) (SProd.sprod (Metric. …
    -/
    exact ⟨⟨x, y⟩, by simpa only [Set.mem_prod] using ⟨⟨hAf x_in_s, hAg y_in_t⟩, xy_eq⟩⟩
    /-
      🎉 no goals
    -/


