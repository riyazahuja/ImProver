theorem fg_trans (h1 : (adjoin R s).toSubmodule.FG) (h2 : (adjoin (adjoin R s) t).toSubmodule.FG) :
    (adjoin R (s ∪ t)).toSubmodule.FG := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s t : Set A
    h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
    h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
    ⊢ (Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t))).FG
  -/
  rcases fg_def.1 h1 with ⟨p, hp, hp'⟩
  /-
    case intro.intro
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s t : Set A
    h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
    h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
    p : Set A
    hp : p.Finite
    hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
    ⊢ (Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t))).FG
  -/
  rcases fg_def.1 h2 with ⟨q, hq, hq'⟩
  /-
    case intro.intro.intro.intro
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    s t : Set A
    h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
    h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
    p : Set A
    hp : p.Finite
    hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
    q : Set A
    hq : q.Finite
    hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
    ⊢ (Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t))).FG
  -/
  refine fg_def.2 ⟨p * q, hp.mul hq, le_antisymm ?_ ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      ⊢ LE.le (Submodule.span R (HMul.hMul p q)) (Subalgebra.toSubmodule (Algebra.ad …
    -/
  · rw [span_le, Set.mul_subset_iff]
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      ⊢ ∀ (x : A), Membership.mem p x → ∀ (y : A), Membership.mem q y → Membership.m …
    -/
    intro x hx y hy
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      x : A
      hx : Membership.mem p x
      y : A
      hy : Membership.mem q y
      ⊢ Membership.mem (↑(Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t) …
    -/
    change x * y ∈ adjoin R (s ∪ t)
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      x : A
      hx : Membership.mem p x
      y : A
      hy : Membership.mem q y
      ⊢ Membership.mem (Algebra.adjoin R (Union.union s t)) (HMul.hMul x y)
    -/
    refine Subalgebra.mul_mem _ ?_ ?_
    · have : x ∈ Subalgebra.toSubmodule (adjoin R s) := by
        rw [← hp']
        exact subset_span hx
      /-
        case intro.intro.intro.intro.refine_1.refine_1
        R : Type u
        A : Type v
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring A
        inst✝ : Algebra R A
        s t : Set A
        h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
        h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
        p : Set A
        hp : p.Finite
        hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
        q : Set A
        hq : q.Finite
        hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
        x : A
        hx : Membership.mem p x
        y : A
        hy : Membership.mem q y
        this : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R s)) x
        ⊢ Membership.mem (Algebra.adjoin R (Union.union s t)) x
      -/
      exact adjoin_mono Set.subset_union_left this
      /-
        🎉 no goals
      -/
    have : y ∈ Subalgebra.toSubmodule (adjoin (adjoin R s) t) := by
      rw [← hq']
      exact subset_span hy
    /-
      case intro.intro.intro.intro.refine_1.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      x : A
      hx : Membership.mem p x
      y : A
      hy : Membership.mem q y
      this : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x = …
      ⊢ Membership.mem (Algebra.adjoin R (Union.union s t)) y
    -/
    change y ∈ adjoin R (s ∪ t)
    /-
      case intro.intro.intro.intro.refine_1.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      x : A
      hx : Membership.mem p x
      y : A
      hy : Membership.mem q y
      this : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x = …
      ⊢ Membership.mem (Algebra.adjoin R (Union.union s t)) y
    -/
    rwa [adjoin_union_eq_adjoin_adjoin]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      ⊢ LE.le (Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t))) (Submodu …
    -/
  · intro r hr
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      r : A
      hr : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R (Union.union s t …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) r
    -/
    change r ∈ adjoin R (s ∪ t) at hr
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      r : A
      hr : Membership.mem (Algebra.adjoin R (Union.union s t)) r
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) r
    -/
    rw [adjoin_union_eq_adjoin_adjoin] at hr
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      r : A
      hr : Membership.mem (Subalgebra.restrictScalars R (Algebra.adjoin (Subtype fun …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) r
    -/
    change r ∈ Subalgebra.toSubmodule (adjoin (adjoin R s) t) at hr
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      r : A
      hr : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x =>  …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) r
    -/
    rw [← hq', ← Set.image_id q, Finsupp.mem_span_image_iff_linearCombination (adjoin R s)] at hr
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      r : A
      hr : Exists fun l => And (Membership.mem (Finsupp.supported (Subtype fun x =>  …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) r
    -/
    rcases hr with ⟨l, hlq, rfl⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) ((Finsupp.linearCombinatio …
    -/
    have := @Finsupp.linearCombination_apply A A (adjoin R s)
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R s …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) ((Finsupp.linearCombinatio …
    -/
    rw [this, Finsupp.sum]
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R s …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (l.support.sum fun a => HS …
    -/
    refine sum_mem ?_
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R s …
      ⊢ ∀ (c : A), Membership.mem l.support c → Membership.mem (Submodule.span R (HM …
    -/
    intro z hz
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R s …
      z : A
      hz : Membership.mem l.support z
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HSMul.hSMul (l z) (_root_ …
    -/
    change (l z).1 * _ ∈ _
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R s …
      z : A
      hz : Membership.mem l.support z
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (↑(l z)) (_root …
    -/
    have : (l z).1 ∈ Subalgebra.toSubmodule (adjoin R s) := (l z).2
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      this : Membership.mem (Subalgebra.toSubmodule (Algebra.adjoin R s)) ↑(l z)
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (↑(l z)) (_root …
    -/
    rw [← hp', ← Set.image_id p, Finsupp.mem_span_image_iff_linearCombination R] at this
    /-
      case intro.intro.intro.intro.refine_2.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      this : Exists fun l_1 => And (Membership.mem (Finsupp.supported R R p) l_1) (E …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (↑(l z)) (_root …
    -/
    rcases this with ⟨l2, hlp, hl⟩
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R s …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq ((Finsupp.linearCombination R _root_.id) l2) ↑(l z)
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (↑(l z)) (_root …
    -/
    have := @Finsupp.linearCombination_apply A A R
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq ((Finsupp.linearCombination R _root_.id) l2) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (↑(l z)) (_root …
    -/
    rw [this] at hl
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (↑(l z)) (_root …
    -/
    rw [← hl, Finsupp.sum_mul]
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (l2.sum fun a c => HMul.hM …
    -/
    refine sum_mem ?_
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      ⊢ ∀ (c : A), Membership.mem l2.support c → Membership.mem (Submodule.span R (H …
    -/
    intro t ht
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t✝ : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      t : A
      ht : Membership.mem l2.support t
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) ((fun a c => HMul.hMul (HS …
    -/
    change _ * _ ∈ _
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t✝ : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      t : A
      ht : Membership.mem l2.support t
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (HSMul.hSMul (l …
    -/
    rw [smul_mul_assoc]
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t✝ : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      t : A
      ht : Membership.mem l2.support t
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HSMul.hSMul (l2 t) (HMul. …
    -/
    refine smul_mem _ _ ?_
    /-
      case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      s t✝ : Set A
      h1 : (Subalgebra.toSubmodule (Algebra.adjoin R s)).FG
      h2 : (Subalgebra.toSubmodule (Algebra.adjoin (Subtype fun x => Membership.mem  …
      p : Set A
      hp : p.Finite
      hp' : Eq (Submodule.span R p) (Subalgebra.toSubmodule (Algebra.adjoin R s))
      q : Set A
      hq : q.Finite
      hq' : Eq (Submodule.span (Subtype fun x => Membership.mem (Algebra.adjoin R s) …
      l : Finsupp A (Subtype fun x => Membership.mem (Algebra.adjoin R s) x)
      hlq : Membership.mem (Finsupp.supported (Subtype fun x => Membership.mem (Alge …
      this✝ : ∀ [inst : Semiring (Subtype fun x => Membership.mem (Algebra.adjoin R  …
      z : A
      hz : Membership.mem l.support z
      l2 : Finsupp A R
      hlp : Membership.mem (Finsupp.supported R R p) l2
      hl : Eq (l2.sum fun i a => HSMul.hSMul a (_root_.id i)) ↑(l z)
      this : ∀ [inst : Semiring R] [inst_1 : AddCommMonoid A] [inst_2 : Module R A]  …
      t : A
      ht : Membership.mem l2.support t
      ⊢ Membership.mem (Submodule.span R (HMul.hMul p q)) (HMul.hMul (_root_.id t) ( …
    -/
    exact subset_span ⟨t, hlp ht, z, hlq hz, rfl⟩
    /-
      🎉 no goals
    -/


/-- A subalgebra `S` is finitely generated if there exists `t : Finset A` such that
`Algebra.adjoin R t = S`. -/
def FG (S : Subalgebra R A) : Prop :=
  ∃ t : Finset A, Algebra.adjoin R ↑t = S


theorem fg_adjoin_finset (s : Finset A) : (Algebra.adjoin R (↑s : Set A)).FG :=
  ⟨s, rfl⟩


theorem fg_def {S : Subalgebra R A} : S.FG ↔ ∃ t : Set A, Set.Finite t ∧ Algebra.adjoin R t = S :=
  Iff.symm Set.exists_finite_iff_finset


theorem fg_bot : (⊥ : Subalgebra R A).FG :=
  ⟨∅, Finset.coe_empty ▸ Algebra.adjoin_empty R A⟩


theorem fg_of_fg_toSubmodule {S : Subalgebra R A} : S.toSubmodule.FG → S.FG :=
  fun ⟨t, ht⟩ ↦ ⟨t, le_antisymm
    (Algebra.adjoin_le fun x hx ↦ show x ∈ Subalgebra.toSubmodule S from ht ▸ subset_span hx) <|
    show Subalgebra.toSubmodule S ≤ Subalgebra.toSubmodule (Algebra.adjoin R ↑t) from fun x hx ↦
      span_le.mpr (fun _ hx ↦ Algebra.subset_adjoin hx)
        (show x ∈ span R ↑t by
          /-
            R : Type u
            A : Type v
            inst✝² : CommSemiring R
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Subalgebra R A
            x✝ : (Subalgebra.toSubmodule S).FG
            t : Finset A
            ht : Eq (Submodule.span R ↑t) (Subalgebra.toSubmodule S)
            x : A
            hx : Membership.mem (Subalgebra.toSubmodule S) x
            ⊢ Membership.mem (Submodule.span R ↑t) x
          -/
          rw [ht]
          /-
            R : Type u
            A : Type v
            inst✝² : CommSemiring R
            inst✝¹ : Semiring A
            inst✝ : Algebra R A
            S : Subalgebra R A
            x✝ : (Subalgebra.toSubmodule S).FG
            t : Finset A
            ht : Eq (Submodule.span R ↑t) (Subalgebra.toSubmodule S)
            x : A
            hx : Membership.mem (Subalgebra.toSubmodule S) x
            ⊢ Membership.mem (Subalgebra.toSubmodule S) x
          -/
          exact hx)⟩
          /-
            🎉 no goals
          -/


theorem fg_of_noetherian [IsNoetherian R A] (S : Subalgebra R A) : S.FG :=
  fg_of_fg_toSubmodule (IsNoetherian.noetherian (Subalgebra.toSubmodule S))


theorem fg_of_submodule_fg (h : (⊤ : Submodule R A).FG) : (⊤ : Subalgebra R A).FG :=
  let ⟨s, hs⟩ := h
  ⟨s, toSubmodule.injective <| by
    /-
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      h : Top.top.FG
      s : Finset A
      hs : Eq (Submodule.span R ↑s) Top.top
      ⊢ Eq (Subalgebra.toSubmodule (Algebra.adjoin R ↑s)) (Subalgebra.toSubmodule To …
    -/
    rw [Algebra.top_toSubmodule, eq_top_iff, ← hs, span_le]
    /-
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      h : Top.top.FG
      s : Finset A
      hs : Eq (Submodule.span R ↑s) Top.top
      ⊢ HasSubset.Subset ↑s ↑(Subalgebra.toSubmodule (Algebra.adjoin R ↑s))
    -/
    exact Algebra.subset_adjoin⟩
    /-
      🎉 no goals
    -/


theorem FG.prod {S : Subalgebra R A} {T : Subalgebra R B} (hS : S.FG) (hT : T.FG) :
    (S.prod T).FG := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    S : Subalgebra R A
    T : Subalgebra R B
    hS : S.FG
    hT : T.FG
    ⊢ (S.prod T).FG
  -/
  obtain ⟨s, hs⟩ := fg_def.1 hS
  /-
    case intro
    R : Type u
    A : Type v
    B : Type w
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    S : Subalgebra R A
    T : Subalgebra R B
    hS : S.FG
    hT : T.FG
    s : Set A
    hs : And s.Finite (Eq (Algebra.adjoin R s) S)
    ⊢ (S.prod T).FG
  -/
  obtain ⟨t, ht⟩ := fg_def.1 hT
  /-
    case intro.intro
    R : Type u
    A : Type v
    B : Type w
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    S : Subalgebra R A
    T : Subalgebra R B
    hS : S.FG
    hT : T.FG
    s : Set A
    hs : And s.Finite (Eq (Algebra.adjoin R s) S)
    t : Set B
    ht : And t.Finite (Eq (Algebra.adjoin R t) T)
    ⊢ (S.prod T).FG
  -/
  rw [← hs.2, ← ht.2]
  exact fg_def.2 ⟨LinearMap.inl R A B '' (s ∪ {1}) ∪ LinearMap.inr R A B '' (t ∪ {1}),
    Set.Finite.union (Set.Finite.image _ (Set.Finite.union hs.1 (Set.finite_singleton _)))
      (Set.Finite.image _ (Set.Finite.union ht.1 (Set.finite_singleton _))),
    Algebra.adjoin_inl_union_inr_eq_prod R s t⟩


theorem FG.map {S : Subalgebra R A} (f : A →ₐ[R] B) (hs : S.FG) : (S.map f).FG :=
  let ⟨s, hs⟩ := hs
                 /-
                   R : Type u
                   A : Type v
                   B : Type w
                   inst✝⁴ : CommSemiring R
                   inst✝³ : Semiring A
                   inst✝² : Algebra R A
                   inst✝¹ : Semiring B
                   inst✝ : Algebra R B
                   S : Subalgebra R A
                   f : AlgHom R A B
                   hs✝ : S.FG
                   s : Finset A
                   hs : Eq (Algebra.adjoin R ↑s) S
                   ⊢ Eq (Algebra.adjoin R ↑(Finset.image (⇑f) s)) (Subalgebra.map f S)
                 -/
  ⟨s.image f, by rw [Finset.coe_image, Algebra.adjoin_image, hs]⟩
                 /-
                   🎉 no goals
                 -/


theorem fg_of_fg_map (S : Subalgebra R A) (f : A →ₐ[R] B) (hf : Function.Injective f)
    (hs : (S.map f).FG) : S.FG :=
  let ⟨s, hs⟩ := hs
  ⟨s.preimage f fun _ _ _ _ h ↦ hf h,
    map_injective hf <| by
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Semiring B
        inst✝ : Algebra R B
        S : Subalgebra R A
        f : AlgHom R A B
        hf : Function.Injective ⇑f
        hs✝ : (Subalgebra.map f S).FG
        s : Finset B
        hs : Eq (Algebra.adjoin R ↑s) (Subalgebra.map f S)
        ⊢ Eq (Subalgebra.map f (Algebra.adjoin R ↑(s.preimage ⇑f ⋯))) (Subalgebra.map  …
      -/
      rw [← Algebra.adjoin_image, Finset.coe_preimage, Set.image_preimage_eq_of_subset, hs]
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Semiring B
        inst✝ : Algebra R B
        S : Subalgebra R A
        f : AlgHom R A B
        hf : Function.Injective ⇑f
        hs✝ : (Subalgebra.map f S).FG
        s : Finset B
        hs : Eq (Algebra.adjoin R ↑s) (Subalgebra.map f S)
        ⊢ HasSubset.Subset (↑s) (Set.range ⇑f)
      -/
      rw [← AlgHom.coe_range, ← Algebra.adjoin_le_iff, hs, ← Algebra.map_top]
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁴ : CommSemiring R
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Semiring B
        inst✝ : Algebra R B
        S : Subalgebra R A
        f : AlgHom R A B
        hf : Function.Injective ⇑f
        hs✝ : (Subalgebra.map f S).FG
        s : Finset B
        hs : Eq (Algebra.adjoin R ↑s) (Subalgebra.map f S)
        ⊢ LE.le (Subalgebra.map f S) (Subalgebra.map f Top.top)
      -/
      exact map_mono le_top⟩
      /-
        🎉 no goals
      -/


theorem fg_top (S : Subalgebra R A) : (⊤ : Subalgebra R S).FG ↔ S.FG :=
  ⟨fun h ↦ by
    /-
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      h : Top.top.FG
      ⊢ S.FG
    -/
    rw [← S.range_val, ← Algebra.map_top]
    /-
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      S : Subalgebra R A
      h : Top.top.FG
      ⊢ (Subalgebra.map S.val Top.top).FG
    -/
    exact FG.map _ h, fun h ↦
    /-
      🎉 no goals
    -/
    fg_of_fg_map _ S.val Subtype.val_injective <| by
      /-
        R : Type u
        A : Type v
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Subalgebra R A
        h : S.FG
        ⊢ (Subalgebra.map S.val Top.top).FG
      -/
      rw [Algebra.map_top, range_val]
      /-
        R : Type u
        A : Type v
        inst✝² : CommSemiring R
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        S : Subalgebra R A
        h : S.FG
        ⊢ S.FG
      -/
      exact h⟩
      /-
        🎉 no goals
      -/


theorem induction_on_adjoin [IsNoetherian R A] (P : Subalgebra R A → Prop) (base : P ⊥)
    (ih : ∀ (S : Subalgebra R A) (x : A), P S → P (Algebra.adjoin R (insert x S)))
    (S : Subalgebra R A) : P S := by
  classical
  obtain ⟨t, rfl⟩ := S.fg_of_noetherian
  refine Finset.induction_on t ?_ ?_
  · simpa using base
  intro x t _ h
  rw [Finset.coe_insert]
  simpa only [Algebra.adjoin_insert_adjoin] using ih _ x h


/-- The image of a Noetherian R-algebra under an R-algebra map is a Noetherian ring. -/
instance AlgHom.isNoetherianRing_range (f : A →ₐ[R] B) [IsNoetherianRing A] :
    IsNoetherianRing f.range :=
  _root_.isNoetherianRing_range f.toRingHom


theorem isNoetherianRing_of_fg {S : Subalgebra R A} (HS : S.FG) [IsNoetherianRing R] :
    IsNoetherianRing S :=
  let ⟨t, ht⟩ := HS
  ht ▸ (Algebra.adjoin_eq_range R (↑t : Set A)).symm ▸ AlgHom.isNoetherianRing_range _


theorem is_noetherian_subring_closure (s : Set R) (hs : s.Finite) :
    IsNoetherianRing (Subring.closure s) :=
  show IsNoetherianRing (subalgebraOfSubring (Subring.closure s)) from
    Algebra.adjoin_int s ▸ isNoetherianRing_of_fg (Subalgebra.fg_def.2 ⟨s, hs, rfl⟩)


