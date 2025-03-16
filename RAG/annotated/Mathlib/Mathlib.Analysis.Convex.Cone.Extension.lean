/-- Induction step in M. Riesz extension theorem. Given a convex cone `s` in a vector space `E`,
a partially defined linear map `f : f.domain → ℝ`, assume that `f` is nonnegative on `f.domain ∩ p`
and `p + s = E`. If `f` is not defined on the whole `E`, then we can extend it to a larger
submodule without breaking the non-negativity condition. -/
theorem step (nonneg : ∀ x : f.domain, (x : E) ∈ s → 0 ≤ f x)
    (dense : ∀ y, ∃ x : f.domain, (x : E) + y ∈ s) (hdom : f.domain ≠ ⊤) :
    ∃ g, f < g ∧ ∀ x : g.domain, (x : E) ∈ s → 0 ≤ g x := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    f : LinearPMap Real E Real
    nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
    dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    hdom : Ne f.domain Top.top
    ⊢ Exists fun g => And (LT.lt f g) (∀ (x : Subtype fun x => Membership.mem g.do …
  -/
  obtain ⟨y, -, hy⟩ : ∃ y ∈ ⊤, y ∉ f.domain := SetLike.exists_of_lt (lt_top_iff_ne_top.2 hdom)
  obtain ⟨c, le_c, c_le⟩ :
      ∃ c, (∀ x : f.domain, -(x : E) - y ∈ s → f x ≤ c) ∧
        ∀ x : f.domain, (x : E) + y ∈ s → c ≤ f x := by
    set Sp := f '' { x : f.domain | (x : E) + y ∈ s }
    set Sn := f '' { x : f.domain | -(x : E) - y ∈ s }
    suffices (upperBounds Sn ∩ lowerBounds Sp).Nonempty by
      simpa only [Sp, Sn, Set.Nonempty, upperBounds, lowerBounds, forall_mem_image] using this
    refine exists_between_of_forall_le (Nonempty.image f ?_) (Nonempty.image f (dense y)) ?_
    · rcases dense (-y) with ⟨x, hx⟩
      rw [← neg_neg x, NegMemClass.coe_neg, ← sub_eq_add_neg] at hx
      exact ⟨_, hx⟩
    rintro a ⟨xn, hxn, rfl⟩ b ⟨xp, hxp, rfl⟩
    have := s.add_mem hxp hxn
    rw [add_assoc, add_sub_cancel, ← sub_eq_add_neg, ← AddSubgroupClass.coe_sub] at this
    replace := nonneg _ this
    rwa [f.map_sub, sub_nonneg] at this
  -- Porting note: removed an unused `have`
  /-
    case intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    f : LinearPMap Real E Real
    nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
    dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    hdom : Ne f.domain Top.top
    y : E
    hy : Not (Membership.mem f.domain y)
    c : Real
    le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
    c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
    ⊢ Exists fun g => And (LT.lt f g) (∀ (x : Subtype fun x => Membership.mem g.do …
  -/
  refine ⟨f.supSpanSingleton y (-c) hy, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      ⊢ LT.lt f (f.supSpanSingleton y (Neg.neg c) hy)
    -/
  · refine lt_iff_le_not_le.2 ⟨f.left_le_sup _ _, fun H => ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      H : LE.le (f.supSpanSingleton y (Neg.neg c) hy) f
      ⊢ False
    -/
    replace H := LinearPMap.domain_mono.monotone H
    /-
      case intro.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      H : LE.le (f.supSpanSingleton y (Neg.neg c) hy).domain f.domain
      ⊢ False
    -/
    rw [LinearPMap.domain_supSpanSingleton, sup_le_iff, span_le, singleton_subset_iff] at H
    /-
      case intro.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      H : And (LE.le f.domain f.domain) (Membership.mem (↑f.domain) y)
      ⊢ False
    -/
    exact hy H.2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      ⊢ ∀ (x : Subtype fun x => Membership.mem (f.supSpanSingleton y (Neg.neg c) hy) …
    -/
  · rintro ⟨z, hz⟩ hzs
    /-
      case intro.intro.intro.intro.refine_2.mk
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      z : E
      hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain z
      hzs : Membership.mem s ↑⟨z, hz⟩
      ⊢ LE.le 0 (↑(f.supSpanSingleton y (Neg.neg c) hy) ⟨z, hz⟩)
    -/
    rcases mem_sup.1 hz with ⟨x, hx, y', hy', rfl⟩
    /-
      case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      x : E
      hx : Membership.mem f.domain x
      y' : E
      hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain y'
      hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
      hzs : Membership.mem s ↑⟨HAdd.hAdd x y', hz⟩
      ⊢ LE.le 0 (↑(f.supSpanSingleton y (Neg.neg c) hy) ⟨HAdd.hAdd x y', hz⟩)
    -/
    rcases mem_span_singleton.1 hy' with ⟨r, rfl⟩
    /-
      case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      x : E
      hx : Membership.mem f.domain x
      r : Real
      hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
      hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
      hzs : Membership.mem s ↑⟨HAdd.hAdd x (HSMul.hSMul r y), hz⟩
      ⊢ LE.le 0 (↑(f.supSpanSingleton y (Neg.neg c) hy) ⟨HAdd.hAdd x (HSMul.hSMul r  …
    -/
    simp only [Subtype.coe_mk] at hzs
    /-
      case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      x : E
      hx : Membership.mem f.domain x
      r : Real
      hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
      hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
      hzs : Membership.mem s (HAdd.hAdd x (HSMul.hSMul r y))
      ⊢ LE.le 0 (↑(f.supSpanSingleton y (Neg.neg c) hy) ⟨HAdd.hAdd x (HSMul.hSMul r  …
    -/
    rw [LinearPMap.supSpanSingleton_apply_mk _ _ _ _ _ hx, smul_neg, ← sub_eq_add_neg, sub_nonneg]
    /-
      case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      hdom : Ne f.domain Top.top
      y : E
      hy : Not (Membership.mem f.domain y)
      c : Real
      le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
      x : E
      hx : Membership.mem f.domain x
      r : Real
      hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
      hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
      hzs : Membership.mem s (HAdd.hAdd x (HSMul.hSMul r y))
      ⊢ LE.le (HSMul.hSMul r c) (↑f ⟨x, hx⟩)
    -/
    rcases lt_trichotomy r 0 with (hr | hr | hr)
    · have : -(r⁻¹ • x) - y ∈ s := by
        rwa [← s.smul_mem_iff (neg_pos.2 hr), smul_sub, smul_neg, neg_smul, neg_neg, smul_smul,
          mul_inv_cancel₀ hr.ne, one_smul, sub_eq_add_neg, neg_smul, neg_neg]
      -- Porting note: added type annotation and `by exact`
      /-
        case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro.inl
        E : Type u_2
        inst✝¹ : AddCommGroup E
        inst✝ : Module Real E
        s : ConvexCone Real E
        f : LinearPMap Real E Real
        nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
        dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
        hdom : Ne f.domain Top.top
        y : E
        hy : Not (Membership.mem f.domain y)
        c : Real
        le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        x : E
        hx : Membership.mem f.domain x
        r : Real
        hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
        hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
        hzs : Membership.mem s (HAdd.hAdd x (HSMul.hSMul r y))
        hr : LT.lt r 0
        this : Membership.mem s (HSub.hSub (Neg.neg (HSMul.hSMul (Inv.inv r) x)) y)
        ⊢ LE.le (HSMul.hSMul r c) (↑f ⟨x, hx⟩)
      -/
      replace : f (r⁻¹ • ⟨x, hx⟩) ≤ c := le_c (r⁻¹ • ⟨x, hx⟩) (by exact this)
      rwa [← mul_le_mul_left (neg_pos.2 hr), neg_mul, neg_mul, neg_le_neg_iff, f.map_smul,
        smul_eq_mul, ← mul_assoc, mul_inv_cancel₀ hr.ne, one_mul] at this
      /-
        case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro.inr.inl
        E : Type u_2
        inst✝¹ : AddCommGroup E
        inst✝ : Module Real E
        s : ConvexCone Real E
        f : LinearPMap Real E Real
        nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
        dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
        hdom : Ne f.domain Top.top
        y : E
        hy : Not (Membership.mem f.domain y)
        c : Real
        le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        x : E
        hx : Membership.mem f.domain x
        r : Real
        hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
        hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
        hzs : Membership.mem s (HAdd.hAdd x (HSMul.hSMul r y))
        hr : Eq r 0
        ⊢ LE.le (HSMul.hSMul r c) (↑f ⟨x, hx⟩)
      -/
    · subst r
      /-
        case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro.inr.inl
        E : Type u_2
        inst✝¹ : AddCommGroup E
        inst✝ : Module Real E
        s : ConvexCone Real E
        f : LinearPMap Real E Real
        nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
        dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
        hdom : Ne f.domain Top.top
        y : E
        hy : Not (Membership.mem f.domain y)
        c : Real
        le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        x : E
        hx : Membership.mem f.domain x
        hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
        hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
        hzs : Membership.mem s (HAdd.hAdd x (HSMul.hSMul 0 y))
        ⊢ LE.le (HSMul.hSMul 0 c) (↑f ⟨x, hx⟩)
      -/
      simp only [zero_smul, add_zero] at hzs ⊢
      /-
        case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro.inr.inl
        E : Type u_2
        inst✝¹ : AddCommGroup E
        inst✝ : Module Real E
        s : ConvexCone Real E
        f : LinearPMap Real E Real
        nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
        dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
        hdom : Ne f.domain Top.top
        y : E
        hy : Not (Membership.mem f.domain y)
        c : Real
        le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        x : E
        hx : Membership.mem f.domain x
        hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
        hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
        hzs : Membership.mem s x
        ⊢ LE.le 0 (↑f ⟨x, hx⟩)
      -/
      apply nonneg
      /-
        case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro.inr.inl.a
        E : Type u_2
        inst✝¹ : AddCommGroup E
        inst✝ : Module Real E
        s : ConvexCone Real E
        f : LinearPMap Real E Real
        nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
        dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
        hdom : Ne f.domain Top.top
        y : E
        hy : Not (Membership.mem f.domain y)
        c : Real
        le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        x : E
        hx : Membership.mem f.domain x
        hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
        hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
        hzs : Membership.mem s x
        ⊢ Membership.mem s ↑⟨x, hx⟩
      -/
      exact hzs
      /-
        🎉 no goals
      -/
    · have : r⁻¹ • x + y ∈ s := by
        rwa [← s.smul_mem_iff hr, smul_add, smul_smul, mul_inv_cancel₀ hr.ne', one_smul]
      -- Porting note: added type annotation and `by exact`
      /-
        case intro.intro.intro.intro.refine_2.mk.intro.intro.intro.intro.intro.inr.inr
        E : Type u_2
        inst✝¹ : AddCommGroup E
        inst✝ : Module Real E
        s : ConvexCone Real E
        f : LinearPMap Real E Real
        nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
        dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
        hdom : Ne f.domain Top.top
        y : E
        hy : Not (Membership.mem f.domain y)
        c : Real
        le_c : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        c_le : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s (H …
        x : E
        hx : Membership.mem f.domain x
        r : Real
        hy' : Membership.mem (LinearPMap.mkSpanSingleton y (Neg.neg c) ⋯).domain (HSMu …
        hz : Membership.mem (f.supSpanSingleton y (Neg.neg c) hy).domain (HAdd.hAdd x  …
        hzs : Membership.mem s (HAdd.hAdd x (HSMul.hSMul r y))
        hr : LT.lt 0 r
        this : Membership.mem s (HAdd.hAdd (HSMul.hSMul (Inv.inv r) x) y)
        ⊢ LE.le (HSMul.hSMul r c) (↑f ⟨x, hx⟩)
      -/
      replace : c ≤ f (r⁻¹ • ⟨x, hx⟩) := c_le (r⁻¹ • ⟨x, hx⟩) (by exact this)
      rwa [← mul_le_mul_left hr, f.map_smul, smul_eq_mul, ← mul_assoc, mul_inv_cancel₀ hr.ne',
        one_mul] at this


theorem exists_top (p : E →ₗ.[ℝ] ℝ) (hp_nonneg : ∀ x : p.domain, (x : E) ∈ s → 0 ≤ p x)
    (hp_dense : ∀ y, ∃ x : p.domain, (x : E) + y ∈ s) :
    ∃ q ≥ p, q.domain = ⊤ ∧ ∀ x : q.domain, (x : E) ∈ s → 0 ≤ q x := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    p : LinearPMap Real E Real
    hp_nonneg : ∀ (x : Subtype fun x => Membership.mem p.domain x), Membership.mem …
    hp_dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    ⊢ Exists fun q => And (GE.ge q p) (And (Eq q.domain Top.top) (∀ (x : Subtype f …
  -/
  set S := { p : E →ₗ.[ℝ] ℝ | ∀ x : p.domain, (x : E) ∈ s → 0 ≤ p x }
  have hSc : ∀ c, c ⊆ S → IsChain (· ≤ ·) c → ∀ y ∈ c, ∃ ub ∈ S, ∀ z ∈ c, z ≤ ub := by
    intro c hcs c_chain y hy
    clear hp_nonneg hp_dense p
    have cne : c.Nonempty := ⟨y, hy⟩
    have hcd : DirectedOn (· ≤ ·) c := c_chain.directedOn
    refine ⟨LinearPMap.sSup c hcd, ?_, fun _ ↦ LinearPMap.le_sSup hcd⟩
    rintro ⟨x, hx⟩ hxs
    have hdir : DirectedOn (· ≤ ·) (LinearPMap.domain '' c) :=
      directedOn_image.2 (hcd.mono LinearPMap.domain_mono.monotone)
    rcases (mem_sSup_of_directed (cne.image _) hdir).1 hx with ⟨_, ⟨f, hfc, rfl⟩, hfx⟩
    have : f ≤ LinearPMap.sSup c hcd := LinearPMap.le_sSup _ hfc
    convert ← hcs hfc ⟨x, hfx⟩ hxs using 1
    exact this.2 rfl
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    p : LinearPMap Real E Real
    hp_nonneg : ∀ (x : Subtype fun x => Membership.mem p.domain x), Membership.mem …
    hp_dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    S : Set (LinearPMap Real E Real) := setOf fun p => ∀ (x : Subtype fun x => Mem …
    hSc : ∀ (c : Set (LinearPMap Real E Real)), HasSubset.Subset c S → IsChain (fu …
    ⊢ Exists fun q => And (GE.ge q p) (And (Eq q.domain Top.top) (∀ (x : Subtype f …
  -/
  obtain ⟨q, hpq, hqs, hq⟩ := zorn_le_nonempty₀ S hSc p hp_nonneg
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    p : LinearPMap Real E Real
    hp_nonneg : ∀ (x : Subtype fun x => Membership.mem p.domain x), Membership.mem …
    hp_dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    S : Set (LinearPMap Real E Real) := setOf fun p => ∀ (x : Subtype fun x => Mem …
    hSc : ∀ (c : Set (LinearPMap Real E Real)), HasSubset.Subset c S → IsChain (fu …
    q : LinearPMap Real E Real
    hpq : LE.le p q
    hqs : Membership.mem S q
    hq : ∀ ⦃y : LinearPMap Real E Real⦄, (fun x => Membership.mem S x) y → LE.le q …
    ⊢ Exists fun q => And (GE.ge q p) (And (Eq q.domain Top.top) (∀ (x : Subtype f …
  -/
  refine ⟨q, hpq, ?_, hqs⟩
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    p : LinearPMap Real E Real
    hp_nonneg : ∀ (x : Subtype fun x => Membership.mem p.domain x), Membership.mem …
    hp_dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    S : Set (LinearPMap Real E Real) := setOf fun p => ∀ (x : Subtype fun x => Mem …
    hSc : ∀ (c : Set (LinearPMap Real E Real)), HasSubset.Subset c S → IsChain (fu …
    q : LinearPMap Real E Real
    hpq : LE.le p q
    hqs : Membership.mem S q
    hq : ∀ ⦃y : LinearPMap Real E Real⦄, (fun x => Membership.mem S x) y → LE.le q …
    ⊢ Eq q.domain Top.top
  -/
  contrapose! hq
  have hqd : ∀ y, ∃ x : q.domain, (x : E) + y ∈ s := fun y ↦
    let ⟨x, hx⟩ := hp_dense y
    ⟨Submodule.inclusion hpq.left x, hx⟩
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    p : LinearPMap Real E Real
    hp_nonneg : ∀ (x : Subtype fun x => Membership.mem p.domain x), Membership.mem …
    hp_dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    S : Set (LinearPMap Real E Real) := setOf fun p => ∀ (x : Subtype fun x => Mem …
    hSc : ∀ (c : Set (LinearPMap Real E Real)), HasSubset.Subset c S → IsChain (fu …
    q : LinearPMap Real E Real
    hpq : LE.le p q
    hqs : Membership.mem S q
    hq : Ne q.domain Top.top
    hqd : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    ⊢ Exists fun ⦃y⦄ => And (Membership.mem S y) (And (LE.le q y) (Not (LE.le y q)))
  -/
  rcases step s q hqs hqd hq with ⟨r, hqr, hr⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    p : LinearPMap Real E Real
    hp_nonneg : ∀ (x : Subtype fun x => Membership.mem p.domain x), Membership.mem …
    hp_dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    S : Set (LinearPMap Real E Real) := setOf fun p => ∀ (x : Subtype fun x => Mem …
    hSc : ∀ (c : Set (LinearPMap Real E Real)), HasSubset.Subset c S → IsChain (fu …
    q : LinearPMap Real E Real
    hpq : LE.le p q
    hqs : Membership.mem S q
    hq : Ne q.domain Top.top
    hqd : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    r : LinearPMap Real E Real
    hqr : LT.lt q r
    hr : ∀ (x : Subtype fun x => Membership.mem r.domain x), Membership.mem s ↑x → …
    ⊢ Exists fun ⦃y⦄ => And (Membership.mem S y) (And (LE.le q y) (Not (LE.le y q)))
  -/
  exact ⟨r, hr, hqr.le, fun hrq ↦ hqr.ne' <| hrq.antisymm hqr.le⟩
  /-
    🎉 no goals
  -/


/-- M. **Riesz extension theorem**: given a convex cone `s` in a vector space `E`, a submodule `p`,
and a linear `f : p → ℝ`, assume that `f` is nonnegative on `p ∩ s` and `p + s = E`. Then
there exists a globally defined linear function `g : E → ℝ` that agrees with `f` on `p`,
and is nonnegative on `s`. -/
theorem riesz_extension (s : ConvexCone ℝ E) (f : E →ₗ.[ℝ] ℝ)
    (nonneg : ∀ x : f.domain, (x : E) ∈ s → 0 ≤ f x)
    (dense : ∀ y, ∃ x : f.domain, (x : E) + y ∈ s) :
    ∃ g : E →ₗ[ℝ] ℝ, (∀ x : f.domain, g x = f x) ∧ ∀ x ∈ s, 0 ≤ g x := by
  rcases RieszExtension.exists_top s f nonneg dense
    with ⟨⟨g_dom, g⟩, ⟨-, hfg⟩, rfl : g_dom = ⊤, hgs⟩
  /-
    case intro.mk.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : ConvexCone Real E
    f : LinearPMap Real E Real
    nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
    dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
    g : LinearMap (RingHom.id Real) (Subtype fun x => Membership.mem Top.top x) Real
    hfg : ∀ ⦃x : Subtype fun x => Membership.mem f.domain x⦄ ⦃y : Subtype fun x => …
    hgs : ∀ (x : Subtype fun x => Membership.mem { domain := Top.top, toFun := g } …
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem f.domain x), Eq  …
  -/
  refine ⟨g.comp (LinearMap.id.codRestrict ⊤ fun _ ↦ trivial), ?_, ?_⟩
    /-
      case intro.mk.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      g : LinearMap (RingHom.id Real) (Subtype fun x => Membership.mem Top.top x) Real
      hfg : ∀ ⦃x : Subtype fun x => Membership.mem f.domain x⦄ ⦃y : Subtype fun x => …
      hgs : ∀ (x : Subtype fun x => Membership.mem { domain := Top.top, toFun := g } …
      ⊢ ∀ (x : Subtype fun x => Membership.mem f.domain x), Eq ((g.comp (LinearMap.c …
    -/
  · exact fun x => (hfg rfl).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : ConvexCone Real E
      f : LinearPMap Real E Real
      nonneg : ∀ (x : Subtype fun x => Membership.mem f.domain x), Membership.mem s  …
      dense : ∀ (y : E), Exists fun x => Membership.mem s (HAdd.hAdd (↑x) y)
      g : LinearMap (RingHom.id Real) (Subtype fun x => Membership.mem Top.top x) Real
      hfg : ∀ ⦃x : Subtype fun x => Membership.mem f.domain x⦄ ⦃y : Subtype fun x => …
      hgs : ∀ (x : Subtype fun x => Membership.mem { domain := Top.top, toFun := g } …
      ⊢ ∀ (x : E), Membership.mem s x → LE.le 0 ((g.comp (LinearMap.codRestrict Top. …
    -/
  · exact fun x hx => hgs ⟨x, _⟩ hx
    /-
      🎉 no goals
    -/


/-- **Hahn-Banach theorem**: if `N : E → ℝ` is a sublinear map, `f` is a linear map
defined on a subspace of `E`, and `f x ≤ N x` for all `x` in the domain of `f`,
then `f` can be extended to the whole space to a linear map `g` such that `g x ≤ N x`
for all `x`. -/
theorem exists_extension_of_le_sublinear (f : E →ₗ.[ℝ] ℝ) (N : E → ℝ)
    (N_hom : ∀ c : ℝ, 0 < c → ∀ x, N (c • x) = c * N x) (N_add : ∀ x y, N (x + y) ≤ N x + N y)
    (hf : ∀ x : f.domain, f x ≤ N x) :
    ∃ g : E →ₗ[ℝ] ℝ, (∀ x : f.domain, g x = f x) ∧ ∀ x, g x ≤ N x := by
  let s : ConvexCone ℝ (E × ℝ) :=
    { carrier := { p : E × ℝ | N p.1 ≤ p.2 }
      smul_mem' := fun c hc p hp =>
        calc
          N (c • p.1) = c * N p.1 := N_hom c hc p.1
          _ ≤ c * p.2 := mul_le_mul_of_nonneg_left hp hc.le
      add_mem' := fun x hx y hy => (N_add _ _).trans (add_le_add hx hy) }
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    f : LinearPMap Real E Real
    N : E → Real
    N_hom : ∀ (c : Real), LT.lt 0 c → ∀ (x : E), Eq (N (HSMul.hSMul c x)) (HMul.hM …
    N_add : ∀ (x y : E), LE.le (N (HAdd.hAdd x y)) (HAdd.hAdd (N x) (N y))
    hf : ∀ (x : Subtype fun x => Membership.mem f.domain x), LE.le (↑f x) (N ↑x)
    s : ConvexCone Real (Prod E Real) := { carrier := setOf fun p => LE.le (N p.1) …
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem f.domain x), Eq  …
  -/
  set f' := (-f).coprod (LinearMap.id.toPMap ⊤)
  have hf'_nonneg : ∀ x : f'.domain, x.1 ∈ s → 0 ≤ f' x := fun x (hx : N x.1.1 ≤ x.1.2) ↦ by
    simpa [f'] using le_trans (hf ⟨x.1.1, x.2.1⟩) hx
  have hf'_dense : ∀ y : E × ℝ, ∃ x : f'.domain, ↑x + y ∈ s := by
    rintro ⟨x, y⟩
    refine ⟨⟨(0, N x - y), ⟨f.domain.zero_mem, trivial⟩⟩, ?_⟩
    simp only [s, ConvexCone.mem_mk, mem_setOf_eq, Prod.fst_add, Prod.snd_add, zero_add,
      sub_add_cancel, le_rfl]
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    f : LinearPMap Real E Real
    N : E → Real
    N_hom : ∀ (c : Real), LT.lt 0 c → ∀ (x : E), Eq (N (HSMul.hSMul c x)) (HMul.hM …
    N_add : ∀ (x y : E), LE.le (N (HAdd.hAdd x y)) (HAdd.hAdd (N x) (N y))
    hf : ∀ (x : Subtype fun x => Membership.mem f.domain x), LE.le (↑f x) (N ↑x)
    s : ConvexCone Real (Prod E Real) := { carrier := setOf fun p => LE.le (N p.1) …
    f' : LinearPMap Real (Prod E Real) Real := (Neg.neg f).coprod (LinearMap.id.to …
    hf'_nonneg : ∀ (x : Subtype fun x => Membership.mem f'.domain x), Membership.m …
    hf'_dense : ∀ (y : Prod E Real), Exists fun x => Membership.mem s (HAdd.hAdd ( …
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem f.domain x), Eq  …
  -/
  obtain ⟨g, g_eq, g_nonneg⟩ := riesz_extension s f' hf'_nonneg hf'_dense
  replace g_eq : ∀ (x : f.domain) (y : ℝ), g (x, y) = y - f x := fun x y ↦
    (g_eq ⟨(x, y), ⟨x.2, trivial⟩⟩).trans (sub_eq_neg_add _ _).symm
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    f : LinearPMap Real E Real
    N : E → Real
    N_hom : ∀ (c : Real), LT.lt 0 c → ∀ (x : E), Eq (N (HSMul.hSMul c x)) (HMul.hM …
    N_add : ∀ (x y : E), LE.le (N (HAdd.hAdd x y)) (HAdd.hAdd (N x) (N y))
    hf : ∀ (x : Subtype fun x => Membership.mem f.domain x), LE.le (↑f x) (N ↑x)
    s : ConvexCone Real (Prod E Real) := { carrier := setOf fun p => LE.le (N p.1) …
    f' : LinearPMap Real (Prod E Real) Real := (Neg.neg f).coprod (LinearMap.id.to …
    hf'_nonneg : ∀ (x : Subtype fun x => Membership.mem f'.domain x), Membership.m …
    hf'_dense : ∀ (y : Prod E Real), Exists fun x => Membership.mem s (HAdd.hAdd ( …
    g : LinearMap (RingHom.id Real) (Prod E Real) Real
    g_nonneg : ∀ (x : Prod E Real), Membership.mem s x → LE.le 0 (g x)
    g_eq : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Real), Eq (g {  …
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem f.domain x), Eq  …
  -/
  refine ⟨-g.comp (inl ℝ E ℝ), fun x ↦ ?_, fun x ↦ ?_⟩
    /-
      case intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      f : LinearPMap Real E Real
      N : E → Real
      N_hom : ∀ (c : Real), LT.lt 0 c → ∀ (x : E), Eq (N (HSMul.hSMul c x)) (HMul.hM …
      N_add : ∀ (x y : E), LE.le (N (HAdd.hAdd x y)) (HAdd.hAdd (N x) (N y))
      hf : ∀ (x : Subtype fun x => Membership.mem f.domain x), LE.le (↑f x) (N ↑x)
      s : ConvexCone Real (Prod E Real) := { carrier := setOf fun p => LE.le (N p.1) …
      f' : LinearPMap Real (Prod E Real) Real := (Neg.neg f).coprod (LinearMap.id.to …
      hf'_nonneg : ∀ (x : Subtype fun x => Membership.mem f'.domain x), Membership.m …
      hf'_dense : ∀ (y : Prod E Real), Exists fun x => Membership.mem s (HAdd.hAdd ( …
      g : LinearMap (RingHom.id Real) (Prod E Real) Real
      g_nonneg : ∀ (x : Prod E Real), Membership.mem s x → LE.le 0 (g x)
      g_eq : ∀ (x : Subtype fun x => Membership.mem f.domain x) (y : Real), Eq (g {  …
      x : Subtype fun x => Membership.mem f.domain x
      ⊢ Eq ((Neg.neg (g.comp (LinearMap.inl Real E Real))) ↑x) (↑f x)
    -/
  · simp [g_eq x 0]
    /-
      🎉 no goals
    -/
  · calc -g (x, 0) = g (0, N x) - g (x, N x) := by simp [← map_sub, ← map_neg]
      _ = N x - g (x, N x) := by simpa using g_eq 0 (N x)
      _ ≤ N x := by simpa using g_nonneg ⟨x, N x⟩ (le_refl (N x))

