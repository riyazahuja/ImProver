/-- If 0 → M' → M → M'' → 0 is exact and M' and M'' are
finitely generated then so is M. -/
theorem fg_of_fg_map_of_fg_inf_ker {R M P : Type*} [Ring R] [AddCommGroup M] [Module R M]
    [AddCommGroup P] [Module R P] (f : M →ₗ[R] P) {s : Submodule R M}
    (hs1 : (s.map f).FG)
    (hs2 : (s ⊓ LinearMap.ker f).FG) : s.FG := by
  /-
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    hs1 : (Submodule.map f s).FG
    hs2 : (Min.min s (LinearMap.ker f)).FG
    ⊢ s.FG
  -/
  haveI := Classical.decEq R
  /-
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    hs1 : (Submodule.map f s).FG
    hs2 : (Min.min s (LinearMap.ker f)).FG
    this : DecidableEq R
    ⊢ s.FG
  -/
  haveI := Classical.decEq M
  /-
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    hs1 : (Submodule.map f s).FG
    hs2 : (Min.min s (LinearMap.ker f)).FG
    this✝ : DecidableEq R
    this : DecidableEq M
    ⊢ s.FG
  -/
  haveI := Classical.decEq P
  /-
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    hs1 : (Submodule.map f s).FG
    hs2 : (Min.min s (LinearMap.ker f)).FG
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    ⊢ s.FG
  -/
  cases' hs1 with t1 ht1
  /-
    case intro
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    hs2 : (Min.min s (LinearMap.ker f)).FG
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    ⊢ s.FG
  -/
  cases' hs2 with t2 ht2
  have : ∀ y ∈ t1, ∃ x ∈ s, f x = y := by
    intro y hy
    have : y ∈ s.map f := by
      rw [← ht1]
      exact subset_span hy
    rcases mem_map.1 this with ⟨x, hx1, hx2⟩
    exact ⟨x, hx1, hx2⟩
  have : ∃ g : P → M, ∀ y ∈ t1, g y ∈ s ∧ f (g y) = y := by
    choose g hg1 hg2 using this
    exists fun y => if H : y ∈ t1 then g y H else 0
    intro y H
    constructor
    · simp only [dif_pos H]
      apply hg1
    · simp only [dif_pos H]
      apply hg2
  /-
    case intro.intro
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝³ : DecidableEq R
    this✝² : DecidableEq M
    this✝¹ : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    this✝ : ∀ (y : P), Membership.mem t1 y → Exists fun x => And (Membership.mem s …
    this : Exists fun g => ∀ (y : P), Membership.mem t1 y → And (Membership.mem s  …
    ⊢ s.FG
  -/
  cases' this with g hg
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝² : DecidableEq R
    this✝¹ : DecidableEq M
    this✝ : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    this : ∀ (y : P), Membership.mem t1 y → Exists fun x => And (Membership.mem s  …
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    ⊢ s.FG
  -/
  clear this
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    ⊢ s.FG
  -/
  exists t1.image g ∪ t2
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    ⊢ Eq (Submodule.span R ↑(Union.union (Finset.image g t1) t2)) s
  -/
  rw [Finset.coe_union, span_union, Finset.coe_image]
  /-
    case intro.intro.intro
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    ⊢ Eq (Max.max (Submodule.span R (Set.image g ↑t1)) (Submodule.span R ↑t2)) s
  -/
  apply le_antisymm
    /-
      case intro.intro.intro.a
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      ⊢ LE.le (Max.max (Submodule.span R (Set.image g ↑t1)) (Submodule.span R ↑t2)) s
    -/
  · refine sup_le (span_le.2 <| image_subset_iff.2 ?_) (span_le.2 ?_)
      /-
        case intro.intro.intro.a.refine_1
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        ⊢ HasSubset.Subset (↑t1) (Set.preimage g ↑s)
      -/
    · intro y hy
      /-
        case intro.intro.intro.a.refine_1
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        y : P
        hy : Membership.mem (↑t1) y
        ⊢ Membership.mem (Set.preimage g ↑s) y
      -/
      exact (hg y hy).1
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.a.refine_2
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        ⊢ HasSubset.Subset ↑t2 ↑s
      -/
    · intro x hx
      /-
        case intro.intro.intro.a.refine_2
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem (↑t2) x
        ⊢ Membership.mem (↑s) x
      -/
      have : x ∈ span R t2 := subset_span hx
      /-
        case intro.intro.intro.a.refine_2
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝² : DecidableEq R
        this✝¹ : DecidableEq M
        this✝ : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem (↑t2) x
        this : Membership.mem (Submodule.span R ↑t2) x
        ⊢ Membership.mem (↑s) x
      -/
      rw [ht2] at this
      /-
        case intro.intro.intro.a.refine_2
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝² : DecidableEq R
        this✝¹ : DecidableEq M
        this✝ : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem (↑t2) x
        this : Membership.mem (Min.min s (LinearMap.ker f)) x
        ⊢ Membership.mem (↑s) x
      -/
      exact this.1
      /-
        🎉 no goals
      -/
  /-
    case intro.intro.intro.a
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    ⊢ LE.le s (Max.max (Submodule.span R (Set.image g ↑t1)) (Submodule.span R ↑t2))
  -/
  intro x hx
  have : f x ∈ s.map f := by
    rw [mem_map]
    exact ⟨x, hx, rfl⟩
  /-
    case intro.intro.intro.a
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝² : DecidableEq R
    this✝¹ : DecidableEq M
    this✝ : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    x : M
    hx : Membership.mem s x
    this : Membership.mem (Submodule.map f s) (f x)
    ⊢ Membership.mem (Max.max (Submodule.span R (Set.image g ↑t1)) (Submodule.span …
  -/
  rw [← ht1, ← Set.image_id (t1 : Set P), Finsupp.mem_span_image_iff_linearCombination] at this
  /-
    case intro.intro.intro.a
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝² : DecidableEq R
    this✝¹ : DecidableEq M
    this✝ : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    x : M
    hx : Membership.mem s x
    this : Exists fun l => And (Membership.mem (Finsupp.supported R R ↑t1) l) (Eq  …
    ⊢ Membership.mem (Max.max (Submodule.span R (Set.image g ↑t1)) (Submodule.span …
  -/
  rcases this with ⟨l, hl1, hl2⟩
  refine
    mem_sup.2
      ⟨(linearCombination R id).toFun ((lmapDomain R R g : (P →₀ R) → M →₀ R) l), ?_,
        x - linearCombination R id ((lmapDomain R R g : (P →₀ R) → M →₀ R) l), ?_,
        add_sub_cancel _ _⟩
    /-
      case intro.intro.intro.a.intro.intro.refine_1
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Membership.mem (Submodule.span R (Set.image g ↑t1)) ((Finsupp.linearCombinat …
    -/
  · rw [← Set.image_id (g '' ↑t1), Finsupp.mem_span_image_iff_linearCombination]
    /-
      case intro.intro.intro.a.intro.intro.refine_1
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Exists fun l_1 => And (Membership.mem (Finsupp.supported R R (Set.image g ↑t …
    -/
    refine ⟨_, ?_, rfl⟩
    /-
      case intro.intro.intro.a.intro.intro.refine_1
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Membership.mem (Finsupp.supported R R (Set.image g ↑t1)) ((Finsupp.lmapDomai …
    -/
    haveI : Inhabited P := ⟨0⟩
    /-
      case intro.intro.intro.a.intro.intro.refine_1
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝² : DecidableEq R
      this✝¹ : DecidableEq M
      this✝ : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      this : Inhabited P
      ⊢ Membership.mem (Finsupp.supported R R (Set.image g ↑t1)) ((Finsupp.lmapDomai …
    -/
    rw [← Finsupp.lmapDomain_supported _ _ g, mem_map]
    /-
      case intro.intro.intro.a.intro.intro.refine_1
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝² : DecidableEq R
      this✝¹ : DecidableEq M
      this✝ : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      this : Inhabited P
      ⊢ Exists fun y => And (Membership.mem (Finsupp.supported R R ↑t1) y) (Eq ((Fin …
    -/
    refine ⟨l, hl1, ?_⟩
    /-
      case intro.intro.intro.a.intro.intro.refine_1
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝² : DecidableEq R
      this✝¹ : DecidableEq M
      this✝ : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      this : Inhabited P
      ⊢ Eq ((Finsupp.lmapDomain R R g) l) ((Finsupp.lmapDomain R R g) l)
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.a.intro.intro.refine_2
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    x : M
    hx : Membership.mem s x
    l : Finsupp P R
    hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
    hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
    ⊢ Membership.mem (Submodule.span R ↑t2) (HSub.hSub x ((Finsupp.linearCombinati …
  -/
  rw [ht2, mem_inf]
  /-
    case intro.intro.intro.a.intro.intro.refine_2
    R : Type u_4
    M : Type u_5
    P : Type u_6
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M P
    s : Submodule R M
    this✝¹ : DecidableEq R
    this✝ : DecidableEq M
    this : DecidableEq P
    t1 : Finset P
    ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
    t2 : Finset M
    ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
    g : P → M
    hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
    x : M
    hx : Membership.mem s x
    l : Finsupp P R
    hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
    hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
    ⊢ And (Membership.mem s (HSub.hSub x ((Finsupp.linearCombination R id) ((Finsu …
  -/
  constructor
    /-
      case intro.intro.intro.a.intro.intro.refine_2.left
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Membership.mem s (HSub.hSub x ((Finsupp.linearCombination R id) ((Finsupp.lm …
    -/
  · apply s.sub_mem hx
    /-
      case intro.intro.intro.a.intro.intro.refine_2.left
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Membership.mem s ((Finsupp.linearCombination R id) ((Finsupp.lmapDomain R R  …
    -/
    rw [Finsupp.linearCombination_apply, Finsupp.lmapDomain_apply, Finsupp.sum_mapDomain_index]
      /-
        case intro.intro.intro.a.intro.intro.refine_2.left
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ Membership.mem s (l.sum fun a m => HSMul.hSMul m (id (g a)))
      -/
    · refine s.sum_mem ?_
      /-
        case intro.intro.intro.a.intro.intro.refine_2.left
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ ∀ (c : P), Membership.mem l.support c → Membership.mem s ((fun a m => HSMul. …
      -/
      intro y hy
      /-
        case intro.intro.intro.a.intro.intro.refine_2.left
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        y : P
        hy : Membership.mem l.support y
        ⊢ Membership.mem s ((fun a m => HSMul.hSMul m (id (g a))) y (l y))
      -/
      exact s.smul_mem _ (hg y (hl1 hy)).1
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.a.intro.intro.refine_2.left.h_zero
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ ∀ (b : M), Eq (HSMul.hSMul 0 (id b)) 0
      -/
    · exact zero_smul _
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.a.intro.intro.refine_2.left.h_add
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ ∀ (b : M) (m₁ m₂ : R), Eq (HSMul.hSMul (HAdd.hAdd m₁ m₂) (id b)) (HAdd.hAdd  …
      -/
    · exact fun _ _ _ => add_smul _ _ _
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.a.intro.intro.refine_2.right
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Membership.mem (LinearMap.ker f) (HSub.hSub x ((Finsupp.linearCombination R  …
    -/
  · rw [LinearMap.mem_ker, f.map_sub, ← hl2]
    /-
      case intro.intro.intro.a.intro.intro.refine_2.right
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Eq (HSub.hSub ((Finsupp.linearCombination R id) l) (f ((Finsupp.linearCombin …
    -/
    rw [Finsupp.linearCombination_apply, Finsupp.linearCombination_apply, Finsupp.lmapDomain_apply]
    /-
      case intro.intro.intro.a.intro.intro.refine_2.right
      R : Type u_4
      M : Type u_5
      P : Type u_6
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M P
      s : Submodule R M
      this✝¹ : DecidableEq R
      this✝ : DecidableEq M
      this : DecidableEq P
      t1 : Finset P
      ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
      t2 : Finset M
      ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
      g : P → M
      hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
      x : M
      hx : Membership.mem s x
      l : Finsupp P R
      hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
      ⊢ Eq (HSub.hSub (l.sum fun i a => HSMul.hSMul a (id i)) (f ((Finsupp.mapDomain …
    -/
    rw [Finsupp.sum_mapDomain_index, Finsupp.sum, Finsupp.sum, map_sum]
      /-
        case intro.intro.intro.a.intro.intro.refine_2.right
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ Eq (HSub.hSub (l.support.sum fun a => HSMul.hSMul (l a) (id a)) (l.support.s …
      -/
    · rw [sub_eq_zero]
      /-
        case intro.intro.intro.a.intro.intro.refine_2.right
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ Eq (l.support.sum fun a => HSMul.hSMul (l a) (id a)) (l.support.sum fun x => …
      -/
      refine Finset.sum_congr rfl fun y hy => ?_
      /-
        case intro.intro.intro.a.intro.intro.refine_2.right
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        y : P
        hy : Membership.mem l.support y
        ⊢ Eq (HSMul.hSMul (l y) (id y)) (f (HSMul.hSMul (l y) (id (g y))))
      -/
      unfold id
      /-
        case intro.intro.intro.a.intro.intro.refine_2.right
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        y : P
        hy : Membership.mem l.support y
        ⊢ Eq (HSMul.hSMul (l y) y) (f (HSMul.hSMul (l y) (g y)))
      -/
      rw [f.map_smul, (hg y (hl1 hy)).2]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.a.intro.intro.refine_2.right.h_zero
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ ∀ (b : M), Eq (HSMul.hSMul 0 (id b)) 0
      -/
    · exact zero_smul _
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.a.intro.intro.refine_2.right.h_add
        R : Type u_4
        M : Type u_5
        P : Type u_6
        inst✝⁴ : Ring R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup P
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M P
        s : Submodule R M
        this✝¹ : DecidableEq R
        this✝ : DecidableEq M
        this : DecidableEq P
        t1 : Finset P
        ht1 : Eq (Submodule.span R ↑t1) (Submodule.map f s)
        t2 : Finset M
        ht2 : Eq (Submodule.span R ↑t2) (Min.min s (LinearMap.ker f))
        g : P → M
        hg : ∀ (y : P), Membership.mem t1 y → And (Membership.mem s (g y)) (Eq (f (g y …
        x : M
        hx : Membership.mem s x
        l : Finsupp P R
        hl1 : Membership.mem (Finsupp.supported R R ↑t1) l
        hl2 : Eq ((Finsupp.linearCombination R id) l) (f x)
        ⊢ ∀ (b : M) (m₁ m₂ : R), Eq (HSMul.hSMul (HAdd.hAdd m₁ m₂) (id b)) (HAdd.hAdd  …
      -/
    · exact fun _ _ _ => add_smul _ _ _
      /-
        🎉 no goals
      -/


/-- The kernel of the composition of two linear maps is finitely generated if both kernels are and
the first morphism is surjective. -/
theorem fg_ker_comp {R M N P : Type*} [Ring R] [AddCommGroup M] [Module R M] [AddCommGroup N]
    [Module R N] [AddCommGroup P] [Module R P] (f : M →ₗ[R] N) (g : N →ₗ[R] P)
    (hf1 : (LinearMap.ker f).FG) (hf2 : (LinearMap.ker g).FG)
    (hsur : Function.Surjective f) : (g.comp f).ker.FG := by
  /-
    R : Type u_4
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hf1 : (LinearMap.ker f).FG
    hf2 : (LinearMap.ker g).FG
    hsur : Function.Surjective ⇑f
    ⊢ (LinearMap.ker (g.comp f)).FG
  -/
  rw [LinearMap.ker_comp]
  /-
    R : Type u_4
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hf1 : (LinearMap.ker f).FG
    hf2 : (LinearMap.ker g).FG
    hsur : Function.Surjective ⇑f
    ⊢ (Submodule.comap f (LinearMap.ker g)).FG
  -/
  apply fg_of_fg_map_of_fg_inf_ker f
    /-
      case hs1
      R : Type u_4
      M : Type u_5
      N : Type u_6
      P : Type u_7
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf1 : (LinearMap.ker f).FG
      hf2 : (LinearMap.ker g).FG
      hsur : Function.Surjective ⇑f
      ⊢ (Submodule.map f (Submodule.comap f (LinearMap.ker g))).FG
    -/
  · rwa [Submodule.map_comap_eq, LinearMap.range_eq_top.2 hsur, top_inf_eq]
    /-
      🎉 no goals
    -/
  · rwa [inf_of_le_right (show (LinearMap.ker f) ≤
      (LinearMap.ker g).comap f from comap_mono bot_le)]


instance Module.Finite.finsupp {ι : Type*} [_root_.Finite ι] [Module.Finite R V] :
    Module.Finite R (ι →₀ V) :=
  Module.Finite.equiv (Finsupp.linearEquivFunOnFinite R V ι).symm


