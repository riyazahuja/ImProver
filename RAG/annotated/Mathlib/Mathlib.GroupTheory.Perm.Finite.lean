theorem isConj_of_support_equiv
    (f : { x // x ∈ (σ.support : Set α) } ≃ { x // x ∈ (τ.support : Set α) })
    (hf : ∀ (x : α) (hx : x ∈ (σ.support : Set α)),
      (f ⟨σ x, apply_mem_support.2 hx⟩ : α) = τ ↑(f ⟨x, hx⟩)) :
    IsConj σ τ := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
    hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
    ⊢ IsConj σ τ
  -/
  refine isConj_iff.2 ⟨Equiv.extendSubtype f, ?_⟩
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
    hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
    ⊢ Eq (HMul.hMul (HMul.hMul f.extendSubtype σ) (Inv.inv f.extendSubtype)) τ
  -/
  rw [mul_inv_eq_iff_eq_mul]
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
    hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
    ⊢ Eq (HMul.hMul f.extendSubtype σ) (HMul.hMul τ f.extendSubtype)
  -/
  ext x
  /-
    case H
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
    hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
    x : α
    ⊢ Eq ((HMul.hMul f.extendSubtype σ) x) ((HMul.hMul τ f.extendSubtype) x)
  -/
  simp only [Perm.mul_apply]
  /-
    case H
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
    hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
    x : α
    ⊢ Eq (f.extendSubtype (σ x)) (τ (f.extendSubtype x))
  -/
  by_cases hx : x ∈ σ.support
    /-
      case pos
      α : Type u
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      σ τ : Equiv.Perm α
      f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
      hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
      x : α
      hx : Membership.mem σ.support x
      ⊢ Eq (f.extendSubtype (σ x)) (τ (f.extendSubtype x))
    -/
  · rw [Equiv.extendSubtype_apply_of_mem, Equiv.extendSubtype_apply_of_mem]
      /-
        case pos
        α : Type u
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        σ τ : Equiv.Perm α
        f : Equiv (Subtype fun x => Membership.mem (↑σ.support) x) (Subtype fun x => M …
        hf : ∀ (x : α) (hx : Membership.mem (↑σ.support) x), Eq (↑(f ⟨σ x, ⋯⟩)) (τ ↑(f …
        x : α
        hx : Membership.mem σ.support x
        ⊢ Eq (↑(f ⟨σ x, ?pos.hx✝⟩)) (τ ↑(f ⟨x, ?pos.hx✝⟩))
      -/
    · exact hf x (Finset.mem_coe.2 hx)
      /-
        🎉 no goals
      -/
  · rwa [Classical.not_not.1 ((not_congr mem_support).1 (Equiv.extendSubtype_not_mem f _ _)),
      Classical.not_not.1 ((not_congr mem_support).mp hx)]


theorem perm_inv_on_of_perm_on_finset {s : Finset α} {f : Perm α} (h : ∀ x ∈ s, f x ∈ s) {y : α}
    (hy : y ∈ s) : f⁻¹ y ∈ s := by
  have h0 : ∀ y ∈ s, ∃ (x : _) (hx : x ∈ s), y = (fun i (_ : i ∈ s) => f i) x hx :=
    Finset.surj_on_of_inj_on_of_card_le (fun x hx => (fun i _ => f i) x hx) (fun a ha => h a ha)
      (fun a₁ a₂ ha₁ ha₂ heq => (Equiv.apply_eq_iff_eq f).mp heq) rfl.ge
  /-
    α : Type u
    s : Finset α
    f : Equiv.Perm α
    h : ∀ (x : α), Membership.mem s x → Membership.mem s (f x)
    y : α
    hy : Membership.mem s y
    h0 : ∀ (y : α), Membership.mem s y → Exists fun x => Exists fun hx => Eq y ((f …
    ⊢ Membership.mem s ((Inv.inv f) y)
  -/
  obtain ⟨y2, hy2, heq⟩ := h0 y hy
  /-
    case intro.intro
    α : Type u
    s : Finset α
    f : Equiv.Perm α
    h : ∀ (x : α), Membership.mem s x → Membership.mem s (f x)
    y : α
    hy : Membership.mem s y
    h0 : ∀ (y : α), Membership.mem s y → Exists fun x => Exists fun hx => Eq y ((f …
    y2 : α
    hy2 : Membership.mem s y2
    heq : Eq y ((fun i x => f i) y2 hy2)
    ⊢ Membership.mem s ((Inv.inv f) y)
  -/
  convert hy2
  /-
    case h.e'_5
    α : Type u
    s : Finset α
    f : Equiv.Perm α
    h : ∀ (x : α), Membership.mem s x → Membership.mem s (f x)
    y : α
    hy : Membership.mem s y
    h0 : ∀ (y : α), Membership.mem s y → Exists fun x => Exists fun hx => Eq y ((f …
    y2 : α
    hy2 : Membership.mem s y2
    heq : Eq y ((fun i x => f i) y2 hy2)
    ⊢ Eq ((Inv.inv f) y) y2
  -/
  rw [heq]
  /-
    case h.e'_5
    α : Type u
    s : Finset α
    f : Equiv.Perm α
    h : ∀ (x : α), Membership.mem s x → Membership.mem s (f x)
    y : α
    hy : Membership.mem s y
    h0 : ∀ (y : α), Membership.mem s y → Exists fun x => Exists fun hx => Eq y ((f …
    y2 : α
    hy2 : Membership.mem s y2
    heq : Eq y ((fun i x => f i) y2 hy2)
    ⊢ Eq ((Inv.inv f) ((fun i x => f i) y2 hy2)) y2
  -/
  simp only [inv_apply_self]
  /-
    🎉 no goals
  -/


theorem perm_inv_mapsTo_of_mapsTo (f : Perm α) {s : Set α} [Finite s] (h : Set.MapsTo f s s) :
    Set.MapsTo (f⁻¹ : _) s s := by
  /-
    α : Type u
    f : Equiv.Perm α
    s : Set α
    inst✝ : Finite ↑s
    h : Set.MapsTo (⇑f) s s
    ⊢ Set.MapsTo (⇑(Inv.inv f)) s s
  -/
  cases nonempty_fintype s
  exact fun x hx =>
    Set.mem_toFinset.mp <|
      perm_inv_on_of_perm_on_finset
        (fun a ha => Set.mem_toFinset.mpr (h (Set.mem_toFinset.mp ha)))
        (Set.mem_toFinset.mpr hx)


@[simp]
theorem perm_inv_mapsTo_iff_mapsTo {f : Perm α} {s : Set α} [Finite s] :
    Set.MapsTo (f⁻¹ : _) s s ↔ Set.MapsTo f s s :=
  ⟨perm_inv_mapsTo_of_mapsTo f⁻¹, perm_inv_mapsTo_of_mapsTo f⟩


theorem perm_inv_on_of_perm_on_finite {f : Perm α} {p : α → Prop} [Finite { x // p x }]
    (h : ∀ x, p x → p (f x)) {x : α} (hx : p x) : p (f⁻¹ x) :=
  -- Porting note: relies heavily on the definitions of `Subtype` and `setOf` unfolding to their
  -- underlying predicate.
  have : Finite { x | p x } := ‹_›
  perm_inv_mapsTo_of_mapsTo (s := {x | p x}) f h hx


/-- If the permutation `f` maps `{x // p x}` into itself, then this returns the permutation
  on `{x // p x}` induced by `f`. Note that the `h` hypothesis is weaker than for
  `Equiv.Perm.subtypePerm`. -/
abbrev subtypePermOfFintype (f : Perm α) {p : α → Prop} [Finite { x // p x }]
    (h : ∀ x, p x → p (f x)) : Perm { x // p x } :=
  f.subtypePerm fun x => ⟨h x, fun h₂ => f.inv_apply_self x ▸ perm_inv_on_of_perm_on_finite h h₂⟩


@[simp]
theorem subtypePermOfFintype_apply (f : Perm α) {p : α → Prop} [Finite { x // p x }]
    (h : ∀ x, p x → p (f x)) (x : { x // p x }) : subtypePermOfFintype f h x = ⟨f x, h x x.2⟩ :=
  rfl


theorem subtypePermOfFintype_one (p : α → Prop) [Finite { x // p x }]
    (h : ∀ x, p x → p ((1 : Perm α) x)) : @subtypePermOfFintype α 1 p _ h = 1 :=
  rfl


theorem perm_mapsTo_inl_iff_mapsTo_inr {m n : Type*} [Finite m] [Finite n] (σ : Perm (m ⊕ n)) :
    Set.MapsTo σ (Set.range Sum.inl) (Set.range Sum.inl) ↔
      Set.MapsTo σ (Set.range Sum.inr) (Set.range Sum.inr) := by
  /-
    m : Type u_1
    n : Type u_2
    inst✝¹ : Finite m
    inst✝ : Finite n
    σ : Equiv.Perm (Sum m n)
    ⊢ Iff (Set.MapsTo (⇑σ) (Set.range Sum.inl) (Set.range Sum.inl)) (Set.MapsTo (⇑ …
  -/
  constructor <;>
      /-
        case mp
        m : Type u_1
        n : Type u_2
        inst✝¹ : Finite m
        inst✝ : Finite n
        σ : Equiv.Perm (Sum m n)
        ⊢ Set.MapsTo (⇑σ) (Set.range Sum.inl) (Set.range Sum.inl) → Set.MapsTo (⇑σ) (S …
      -/
    ( intro h
      classical
        rw [← perm_inv_mapsTo_iff_mapsTo] at h
        intro x
        cases' hx : σ x with l r)
    /-
      case mp.inl
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inl) (Set.range Sum.inl)
      x : Sum m n
      l : m
      hx : Eq (σ x) (Sum.inl l)
      ⊢ Membership.mem (Set.range Sum.inr) x → Membership.mem (Set.range Sum.inr) (S …
    -/
  · rintro ⟨a, rfl⟩
    /-
      case mp.inl.intro
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inl) (Set.range Sum.inl)
      l : m
      a : n
      hx : Eq (σ (Sum.inr a)) (Sum.inl l)
      ⊢ Membership.mem (Set.range Sum.inr) (Sum.inl l)
    -/
    obtain ⟨y, hy⟩ := h ⟨l, rfl⟩
    /-
      case mp.inl.intro.intro
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inl) (Set.range Sum.inl)
      l : m
      a : n
      hx : Eq (σ (Sum.inr a)) (Sum.inl l)
      y : m
      hy : Eq (Sum.inl y) ((Inv.inv σ) (Sum.inl l))
      ⊢ Membership.mem (Set.range Sum.inr) (Sum.inl l)
    -/
    rw [← hx, σ.inv_apply_self] at hy
    /-
      case mp.inl.intro.intro
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inl) (Set.range Sum.inl)
      l : m
      a : n
      hx : Eq (σ (Sum.inr a)) (Sum.inl l)
      y : m
      hy : Eq (Sum.inl y) (Sum.inr a)
      ⊢ Membership.mem (Set.range Sum.inr) (Sum.inl l)
    -/
    exact absurd hy Sum.inl_ne_inr
    /-
      🎉 no goals
    -/
    /-
      case mp.inr
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inl) (Set.range Sum.inl)
      x : Sum m n
      r : n
      hx : Eq (σ x) (Sum.inr r)
      ⊢ Membership.mem (Set.range Sum.inr) x → Membership.mem (Set.range Sum.inr) (S …
    -/
  · rintro _; exact ⟨r, rfl⟩
              /-
                🎉 no goals
              -/
    /-
      case mpr.inl
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inr) (Set.range Sum.inr)
      x : Sum m n
      l : m
      hx : Eq (σ x) (Sum.inl l)
      ⊢ Membership.mem (Set.range Sum.inl) x → Membership.mem (Set.range Sum.inl) (S …
    -/
  · rintro _; exact ⟨l, rfl⟩
              /-
                🎉 no goals
              -/
    /-
      case mpr.inr
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inr) (Set.range Sum.inr)
      x : Sum m n
      r : n
      hx : Eq (σ x) (Sum.inr r)
      ⊢ Membership.mem (Set.range Sum.inl) x → Membership.mem (Set.range Sum.inl) (S …
    -/
  · rintro ⟨a, rfl⟩
    /-
      case mpr.inr.intro
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inr) (Set.range Sum.inr)
      r : n
      a : m
      hx : Eq (σ (Sum.inl a)) (Sum.inr r)
      ⊢ Membership.mem (Set.range Sum.inl) (Sum.inr r)
    -/
    obtain ⟨y, hy⟩ := h ⟨r, rfl⟩
    /-
      case mpr.inr.intro.intro
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inr) (Set.range Sum.inr)
      r : n
      a : m
      hx : Eq (σ (Sum.inl a)) (Sum.inr r)
      y : n
      hy : Eq (Sum.inr y) ((Inv.inv σ) (Sum.inr r))
      ⊢ Membership.mem (Set.range Sum.inl) (Sum.inr r)
    -/
    rw [← hx, σ.inv_apply_self] at hy
    /-
      case mpr.inr.intro.intro
      m : Type u_1
      n : Type u_2
      inst✝¹ : Finite m
      inst✝ : Finite n
      σ : Equiv.Perm (Sum m n)
      h : Set.MapsTo (⇑(Inv.inv σ)) (Set.range Sum.inr) (Set.range Sum.inr)
      r : n
      a : m
      hx : Eq (σ (Sum.inl a)) (Sum.inr r)
      y : n
      hy : Eq (Sum.inr y) (Sum.inl a)
      ⊢ Membership.mem (Set.range Sum.inl) (Sum.inr r)
    -/
    exact absurd hy Sum.inr_ne_inl
    /-
      🎉 no goals
    -/


theorem mem_sumCongrHom_range_of_perm_mapsTo_inl {m n : Type*} [Finite m] [Finite n]
    {σ : Perm (m ⊕ n)} (h : Set.MapsTo σ (Set.range Sum.inl) (Set.range Sum.inl)) :
    σ ∈ (sumCongrHom m n).range := by
  classical
    have h1 : ∀ x : m ⊕ n, (∃ a : m, Sum.inl a = x) → ∃ a : m, Sum.inl a = σ x := by
      rintro x ⟨a, ha⟩
      apply h
      rw [← ha]
      exact ⟨a, rfl⟩
    have h3 : ∀ x : m ⊕ n, (∃ b : n, Sum.inr b = x) → ∃ b : n, Sum.inr b = σ x := by
      rintro x ⟨b, hb⟩
      apply (perm_mapsTo_inl_iff_mapsTo_inr σ).mp h
      rw [← hb]
      exact ⟨b, rfl⟩
    let σ₁' := subtypePermOfFintype σ h1
    let σ₂' := subtypePermOfFintype σ h3
    let σ₁ := permCongr (Equiv.ofInjective _ Sum.inl_injective).symm σ₁'
    let σ₂ := permCongr (Equiv.ofInjective _ Sum.inr_injective).symm σ₂'
    rw [MonoidHom.mem_range, Prod.exists]
    use σ₁, σ₂
    rw [Perm.sumCongrHom_apply]
    ext x
    cases' x with a b
    · rw [Equiv.sumCongr_apply, Sum.map_inl, permCongr_apply, Equiv.symm_symm,
        apply_ofInjective_symm Sum.inl_injective]
      rw [ofInjective_apply, Subtype.coe_mk, Subtype.coe_mk]
      -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [subtypePerm_apply]
    · rw [Equiv.sumCongr_apply, Sum.map_inr, permCongr_apply, Equiv.symm_symm,
        apply_ofInjective_symm Sum.inr_injective]
      erw [subtypePerm_apply]
      rw [ofInjective_apply, Subtype.coe_mk, Subtype.coe_mk]


nonrec theorem Disjoint.orderOf {σ τ : Perm α} (hστ : Disjoint σ τ) :
    orderOf (σ * τ) = Nat.lcm (orderOf σ) (orderOf τ) :=
  haveI h : ∀ n : ℕ, (σ * τ) ^ n = 1 ↔ σ ^ n = 1 ∧ τ ^ n = 1 := fun n => by
    /-
      α : Type u
      σ τ : Equiv.Perm α
      hστ : σ.Disjoint τ
      n : Nat
      ⊢ Iff (Eq (HPow.hPow (HMul.hMul σ τ) n) 1) (And (Eq (HPow.hPow σ n) 1) (Eq (HP …
    -/
    rw [hστ.commute.mul_pow, Disjoint.mul_eq_one_iff (hστ.pow_disjoint_pow n n)]
    /-
      🎉 no goals
    -/
  Nat.dvd_antisymm hστ.commute.orderOf_mul_dvd_lcm
    (Nat.lcm_dvd
      (orderOf_dvd_of_pow_eq_one ((h (orderOf (σ * τ))).mp (pow_orderOf_eq_one (σ * τ))).1)
      (orderOf_dvd_of_pow_eq_one ((h (orderOf (σ * τ))).mp (pow_orderOf_eq_one (σ * τ))).2))


theorem Disjoint.extendDomain {p : β → Prop} [DecidablePred p] (f : α ≃ Subtype p)
    {σ τ : Perm α} (h : Disjoint σ τ) : Disjoint (σ.extendDomain f) (τ.extendDomain f) := by
  /-
    α : Type u
    β : Type v
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    σ τ : Equiv.Perm α
    h : σ.Disjoint τ
    ⊢ (σ.extendDomain f).Disjoint (τ.extendDomain f)
  -/
  intro b
  /-
    α : Type u
    β : Type v
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    σ τ : Equiv.Perm α
    h : σ.Disjoint τ
    b : β
    ⊢ Or (Eq ((σ.extendDomain f) b) b) (Eq ((τ.extendDomain f) b) b)
  -/
  by_cases pb : p b
    /-
      case pos
      α : Type u
      β : Type v
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      σ τ : Equiv.Perm α
      h : σ.Disjoint τ
      b : β
      pb : p b
      ⊢ Or (Eq ((σ.extendDomain f) b) b) (Eq ((τ.extendDomain f) b) b)
    -/
  · refine (h (f.symm ⟨b, pb⟩)).imp ?_ ?_ <;>
        /-
          case pos.refine_1
          α : Type u
          β : Type v
          p : β → Prop
          inst✝ : DecidablePred p
          f : Equiv α (Subtype p)
          σ τ : Equiv.Perm α
          h : σ.Disjoint τ
          b : β
          pb : p b
          ⊢ Eq (σ (f.symm ⟨b, pb⟩)) (f.symm ⟨b, pb⟩) → Eq ((σ.extendDomain f) b) b
        -/
        /-
          case pos.refine_1
          α : Type u
          β : Type v
          p : β → Prop
          inst✝ : DecidablePred p
          f : Equiv α (Subtype p)
          σ τ : Equiv.Perm α
          h✝ : σ.Disjoint τ
          b : β
          pb : p b
          h : Eq (σ (f.symm ⟨b, pb⟩)) (f.symm ⟨b, pb⟩)
          ⊢ Eq ((σ.extendDomain f) b) b
        -/
        /-
          🎉 no goals
        -/
        /-
          case pos.refine_2
          α : Type u
          β : Type v
          p : β → Prop
          inst✝ : DecidablePred p
          f : Equiv α (Subtype p)
          σ τ : Equiv.Perm α
          h✝ : σ.Disjoint τ
          b : β
          pb : p b
          h : Eq (τ (f.symm ⟨b, pb⟩)) (f.symm ⟨b, pb⟩)
          ⊢ Eq ((τ.extendDomain f) b) b
        -/
        rw [extendDomain_apply_subtype _ _ pb, h, apply_symm_apply, Subtype.coe_mk]
        /-
          🎉 no goals
        -/
    /-
      case neg
      α : Type u
      β : Type v
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      σ τ : Equiv.Perm α
      h : σ.Disjoint τ
      b : β
      pb : Not (p b)
      ⊢ Or (Eq ((σ.extendDomain f) b) b) (Eq ((τ.extendDomain f) b) b)
    -/
  · left
    /-
      case neg.h
      α : Type u
      β : Type v
      p : β → Prop
      inst✝ : DecidablePred p
      f : Equiv α (Subtype p)
      σ τ : Equiv.Perm α
      h : σ.Disjoint τ
      b : β
      pb : Not (p b)
      ⊢ Eq ((σ.extendDomain f) b) b
    -/
    rw [extendDomain_apply_not_subtype _ _ pb]
    /-
      🎉 no goals
    -/


theorem Disjoint.isConj_mul [Finite α] {σ τ π ρ : Perm α} (hc1 : IsConj σ π)
    (hc2 : IsConj τ ρ) (hd1 : Disjoint σ τ) (hd2 : Disjoint π ρ) : IsConj (σ * τ) (π * ρ) := by
  classical
    cases nonempty_fintype α
    obtain ⟨f, rfl⟩ := isConj_iff.1 hc1
    obtain ⟨g, rfl⟩ := isConj_iff.1 hc2
    have hd1' := coe_inj.2 hd1.support_mul
    have hd2' := coe_inj.2 hd2.support_mul
    rw [coe_union] at *
    have hd1'' := disjoint_coe.2 (disjoint_iff_disjoint_support.1 hd1)
    have hd2'' := disjoint_coe.2 (disjoint_iff_disjoint_support.1 hd2)
    refine isConj_of_support_equiv ?_ ?_
    · refine
          ((Equiv.Set.ofEq hd1').trans (Equiv.Set.union hd1'')).trans
            ((Equiv.sumCongr (subtypeEquiv f fun a => ?_) (subtypeEquiv g fun a => ?_)).trans
              ((Equiv.Set.ofEq hd2').trans (Equiv.Set.union hd2'')).symm) <;>
      · simp only [Set.mem_image, toEmbedding_apply, exists_eq_right, support_conj, coe_map,
          apply_eq_iff_eq]
    · intro x hx
      simp only [trans_apply, symm_trans_apply, Equiv.Set.ofEq_apply, Equiv.Set.ofEq_symm_apply,
        Equiv.sumCongr_apply]
      rw [hd1', Set.mem_union] at hx
      cases' hx with hxσ hxτ
      · rw [mem_coe, mem_support] at hxσ
        rw [Set.union_apply_left, Set.union_apply_left]
        · simp only [subtypeEquiv_apply, Perm.coe_mul, Sum.map_inl, comp_apply,
            Set.union_symm_apply_left, Subtype.coe_mk, apply_eq_iff_eq]
          have h := (hd2 (f x)).resolve_left ?_
          · rw [mul_apply, mul_apply] at h
            rw [h, inv_apply_self, (hd1 x).resolve_left hxσ]
          · rwa [mul_apply, mul_apply, inv_apply_self, apply_eq_iff_eq]
        · rwa [Subtype.coe_mk, mem_coe, mem_support]
        · rwa [Subtype.coe_mk, Perm.mul_apply, (hd1 x).resolve_left hxσ, mem_coe,
            apply_mem_support, mem_support]
      · rw [mem_coe, ← apply_mem_support, mem_support] at hxτ
        rw [Set.union_apply_right, Set.union_apply_right]
        · simp only [subtypeEquiv_apply, Perm.coe_mul, Sum.map_inr, comp_apply,
            Set.union_symm_apply_right, Subtype.coe_mk, apply_eq_iff_eq]
          have h := (hd2 (g (τ x))).resolve_right ?_
          · rw [mul_apply, mul_apply] at h
            rw [inv_apply_self, h, (hd1 (τ x)).resolve_right hxτ]
          · rwa [mul_apply, mul_apply, inv_apply_self, apply_eq_iff_eq]
        · rwa [Subtype.coe_mk, mem_coe, ← apply_mem_support, mem_support]
        · rwa [Subtype.coe_mk, Perm.mul_apply, (hd1 (τ x)).resolve_right hxτ,
            mem_coe, mem_support]


theorem mem_fixedPoints_iff_apply_mem_of_mem_centralizer {g p : Perm α}
    (hp : p ∈ Subgroup.centralizer {g}) {x : α} :
    x ∈ Function.fixedPoints g ↔ p x ∈ Function.fixedPoints g :=  by
  /-
    α : Type u
    g p : Equiv.Perm α
    hp : Membership.mem (Subgroup.centralizer (Singleton.singleton g)) p
    x : α
    ⊢ Iff (Membership.mem (Function.fixedPoints ⇑g) x) (Membership.mem (Function.f …
  -/
  simp only [Subgroup.mem_centralizer_singleton_iff] at hp
  /-
    α : Type u
    g p : Equiv.Perm α
    x : α
    hp : Eq (HMul.hMul p g) (HMul.hMul g p)
    ⊢ Iff (Membership.mem (Function.fixedPoints ⇑g) x) (Membership.mem (Function.f …
  -/
  simp only [Function.mem_fixedPoints_iff]
  /-
    α : Type u
    g p : Equiv.Perm α
    x : α
    hp : Eq (HMul.hMul p g) (HMul.hMul g p)
    ⊢ Iff (Eq (g x) x) (Eq (g (p x)) (p x))
  -/
  rw [← mul_apply, ← hp, mul_apply, EmbeddingLike.apply_eq_iff_eq]
  /-
    🎉 no goals
  -/




lemma disjoint_ofSubtype_of_memFixedPoints_self {g : Perm α}
    (u : Perm (Function.fixedPoints g)) :
    Disjoint (ofSubtype u) g := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    u : Equiv.Perm ↑(Function.fixedPoints ⇑g)
    ⊢ (Equiv.Perm.ofSubtype u).Disjoint g
  -/
  rw [disjoint_iff_eq_or_eq]
  /-
    α : Type u
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    u : Equiv.Perm ↑(Function.fixedPoints ⇑g)
    ⊢ ∀ (x : α), Or (Eq ((Equiv.Perm.ofSubtype u) x) x) (Eq (g x) x)
  -/
  intro x
  /-
    α : Type u
    inst✝ : DecidableEq α
    g : Equiv.Perm α
    u : Equiv.Perm ↑(Function.fixedPoints ⇑g)
    x : α
    ⊢ Or (Eq ((Equiv.Perm.ofSubtype u) x) x) (Eq (g x) x)
  -/
  by_cases hx : x ∈ Function.fixedPoints g
    /-
      case pos
      α : Type u
      inst✝ : DecidableEq α
      g : Equiv.Perm α
      u : Equiv.Perm ↑(Function.fixedPoints ⇑g)
      x : α
      hx : Membership.mem (Function.fixedPoints ⇑g) x
      ⊢ Or (Eq ((Equiv.Perm.ofSubtype u) x) x) (Eq (g x) x)
    -/
  · right; exact hx
           /-
             🎉 no goals
           -/
    /-
      case neg
      α : Type u
      inst✝ : DecidableEq α
      g : Equiv.Perm α
      u : Equiv.Perm ↑(Function.fixedPoints ⇑g)
      x : α
      hx : Not (Membership.mem (Function.fixedPoints ⇑g) x)
      ⊢ Or (Eq ((Equiv.Perm.ofSubtype u) x) x) (Eq (g x) x)
    -/
  · left; rw [ofSubtype_apply_of_not_mem u hx]
          /-
            🎉 no goals
          -/


theorem support_pow_coprime {σ : Perm α} {n : ℕ} (h : Nat.Coprime n (orderOf σ)) :
    (σ ^ n).support = σ.support := by
  /-
    α : Type u
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    n : Nat
    h : n.Coprime (orderOf σ)
    ⊢ Eq (HPow.hPow σ n).support σ.support
  -/
  obtain ⟨m, hm⟩ := exists_pow_eq_self_of_coprime h
  exact
    le_antisymm (support_pow_le σ n)
      (le_trans (ge_of_eq (congr_arg support hm)) (support_pow_le (σ ^ n) m))


