/-- The two-sided ideal of compactly supported functions. -/
def compactlySupported (α γ : Type*) [TopologicalSpace α] [NonUnitalNormedRing γ] :
    TwoSidedIdeal (α →ᵇ γ) :=
  .mk' {z | HasCompactSupport z} .zero .add .neg' .mul_left .mul_right


@[inherit_doc]
scoped[BoundedContinuousFunction] notation
  "C_cb(" α ", " γ ")" => compactlySupported α γ


lemma mem_compactlySupported {f : α →ᵇ γ} :
    f ∈ C_cb(α, γ) ↔ HasCompactSupport f :=
  TwoSidedIdeal.mem_mk' {z : α →ᵇ γ | HasCompactSupport z} .zero .add .neg' .mul_left .mul_right f


lemma exist_norm_eq [c : Nonempty α] {f : α →ᵇ γ} (h : f ∈ C_cb(α, γ)) : ∃ (x : α),
    ‖f x‖ = ‖f‖ := by
  /-
    α : Type u_1
    γ : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : NonUnitalNormedRing γ
    c : Nonempty α
    f : BoundedContinuousFunction α γ
    h : Membership.mem (compactlySupported α γ) f
    ⊢ Exists fun x => Eq (Norm.norm (f x)) (Norm.norm f)
  -/
  by_cases hs : (tsupport f).Nonempty
  · obtain ⟨x, _, hmax⟩ := mem_compactlySupported.mp h |>.exists_isMaxOn hs <|
      (map_continuous f).norm.continuousOn
    /-
      case pos.intro.intro
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      c : Nonempty α
      f : BoundedContinuousFunction α γ
      h : Membership.mem (compactlySupported α γ) f
      hs : (tsupport ⇑f).Nonempty
      x : α
      left✝ : Membership.mem (tsupport ⇑f) x
      hmax : IsMaxOn (fun x => Norm.norm (f x)) (tsupport ⇑f) x
      ⊢ Exists fun x => Eq (Norm.norm (f x)) (Norm.norm f)
    -/
    refine ⟨x, le_antisymm (norm_coe_le_norm f x) (norm_le (norm_nonneg _) |>.mpr fun y ↦ ?_)⟩
    /-
      case pos.intro.intro
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      c : Nonempty α
      f : BoundedContinuousFunction α γ
      h : Membership.mem (compactlySupported α γ) f
      hs : (tsupport ⇑f).Nonempty
      x : α
      left✝ : Membership.mem (tsupport ⇑f) x
      hmax : IsMaxOn (fun x => Norm.norm (f x)) (tsupport ⇑f) x
      y : α
      ⊢ LE.le (Norm.norm (f y)) (Norm.norm (f x))
    -/
    by_cases hy : y ∈ tsupport f
      /-
        case pos
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        c : Nonempty α
        f : BoundedContinuousFunction α γ
        h : Membership.mem (compactlySupported α γ) f
        hs : (tsupport ⇑f).Nonempty
        x : α
        left✝ : Membership.mem (tsupport ⇑f) x
        hmax : IsMaxOn (fun x => Norm.norm (f x)) (tsupport ⇑f) x
        y : α
        hy : Membership.mem (tsupport ⇑f) y
        ⊢ LE.le (Norm.norm (f y)) (Norm.norm (f x))
      -/
    · exact hmax hy
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        c : Nonempty α
        f : BoundedContinuousFunction α γ
        h : Membership.mem (compactlySupported α γ) f
        hs : (tsupport ⇑f).Nonempty
        x : α
        left✝ : Membership.mem (tsupport ⇑f) x
        hmax : IsMaxOn (fun x => Norm.norm (f x)) (tsupport ⇑f) x
        y : α
        hy : Not (Membership.mem (tsupport ⇑f) y)
        ⊢ LE.le (Norm.norm (f y)) (Norm.norm (f x))
      -/
    · simp [image_eq_zero_of_nmem_tsupport hy]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      c : Nonempty α
      f : BoundedContinuousFunction α γ
      h : Membership.mem (compactlySupported α γ) f
      hs : Not (tsupport ⇑f).Nonempty
      ⊢ Exists fun x => Eq (Norm.norm (f x)) (Norm.norm f)
    -/
  · suffices f = 0 by simp [this]
    /-
      case neg
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      c : Nonempty α
      f : BoundedContinuousFunction α γ
      h : Membership.mem (compactlySupported α γ) f
      hs : Not (tsupport ⇑f).Nonempty
      ⊢ Eq f 0
    -/
    rwa [not_nonempty_iff_eq_empty, tsupport_eq_empty_iff, ← coe_zero, ← DFunLike.ext'_iff] at hs
    /-
      🎉 no goals
    -/


theorem norm_lt_iff_of_compactlySupported {f : α →ᵇ γ} (h : f ∈ C_cb(α, γ)) {M : ℝ}
    (M0 : 0 < M) : ‖f‖ < M ↔ ∀ (x : α), ‖f x‖ < M := by
  /-
    α : Type u_1
    γ : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : NonUnitalNormedRing γ
    f : BoundedContinuousFunction α γ
    h : Membership.mem (compactlySupported α γ) f
    M : Real
    M0 : LT.lt 0 M
    ⊢ Iff (LT.lt (Norm.norm f) M) (∀ (x : α), LT.lt (Norm.norm (f x)) M)
  -/
  refine ⟨fun hn x ↦ lt_of_le_of_lt (norm_coe_le_norm f x) hn, ?_⟩
    /-
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      f : BoundedContinuousFunction α γ
      h : Membership.mem (compactlySupported α γ) f
      M : Real
      M0 : LT.lt 0 M
      ⊢ (∀ (x : α), LT.lt (Norm.norm (f x)) M) → LT.lt (Norm.norm f) M
    -/
  · obtain (he | he) := isEmpty_or_nonempty α
      /-
        case inl
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        f : BoundedContinuousFunction α γ
        h : Membership.mem (compactlySupported α γ) f
        M : Real
        M0 : LT.lt 0 M
        he : IsEmpty α
        ⊢ (∀ (x : α), LT.lt (Norm.norm (f x)) M) → LT.lt (Norm.norm f) M
      -/
    · simpa
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        f : BoundedContinuousFunction α γ
        h : Membership.mem (compactlySupported α γ) f
        M : Real
        M0 : LT.lt 0 M
        he : Nonempty α
        ⊢ (∀ (x : α), LT.lt (Norm.norm (f x)) M) → LT.lt (Norm.norm f) M
      -/
    · obtain ⟨x, hx⟩ := exist_norm_eq h
      /-
        case inr.intro
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        f : BoundedContinuousFunction α γ
        h : Membership.mem (compactlySupported α γ) f
        M : Real
        M0 : LT.lt 0 M
        he : Nonempty α
        x : α
        hx : Eq (Norm.norm (f x)) (Norm.norm f)
        ⊢ (∀ (x : α), LT.lt (Norm.norm (f x)) M) → LT.lt (Norm.norm f) M
      -/
      exact fun h ↦ hx ▸ h x
      /-
        🎉 no goals
      -/


theorem norm_lt_iff_of_nonempty_compactlySupported [c : Nonempty α] {f : α →ᵇ γ}
    (h : f ∈ C_cb(α, γ)) {M : ℝ} : ‖f‖ < M ↔ ∀ (x : α), ‖f x‖ < M := by
  /-
    α : Type u_1
    γ : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : NonUnitalNormedRing γ
    c : Nonempty α
    f : BoundedContinuousFunction α γ
    h : Membership.mem (compactlySupported α γ) f
    M : Real
    ⊢ Iff (LT.lt (Norm.norm f) M) (∀ (x : α), LT.lt (Norm.norm (f x)) M)
  -/
  obtain (hM | hM) := lt_or_le 0 M
    /-
      case inl
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      c : Nonempty α
      f : BoundedContinuousFunction α γ
      h : Membership.mem (compactlySupported α γ) f
      M : Real
      hM : LT.lt 0 M
      ⊢ Iff (LT.lt (Norm.norm f) M) (∀ (x : α), LT.lt (Norm.norm (f x)) M)
    -/
  · exact norm_lt_iff_of_compactlySupported h hM
    /-
      🎉 no goals
    -/
  · exact ⟨fun h ↦ False.elim <| (h.trans_le hM).not_le (by positivity),
      fun h ↦ False.elim <| (h (Classical.arbitrary α) |>.trans_le hM).not_le (by positivity)⟩


theorem compactlySupported_eq_top_of_isCompact (h : IsCompact (Set.univ : Set α)) :
    C_cb(α, γ) = ⊤ :=
  eq_top_iff.mpr fun _ _ ↦ h.of_isClosed_subset (isClosed_tsupport _) (subset_univ _)

/- This is intentionally not marked `@[simp]` to prevent Lean looking for a `CompactSpace α`
instance every time it sees `C_cb(α, γ)`. -/

theorem compactlySupported_eq_top [CompactSpace α] : C_cb(α, γ) = ⊤ :=
  compactlySupported_eq_top_of_isCompact CompactSpace.isCompact_univ


theorem compactlySupported_eq_top_iff [Nontrivial γ] :
    C_cb(α, γ) = ⊤ ↔ IsCompact (Set.univ : Set α) := by
  /-
    α : Type u_1
    γ : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : NonUnitalNormedRing γ
    inst✝ : Nontrivial γ
    ⊢ Iff (Eq (compactlySupported α γ) Top.top) (IsCompact Set.univ)
  -/
  refine ⟨fun h ↦ ?_, compactlySupported_eq_top_of_isCompact⟩
  /-
    α : Type u_1
    γ : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : NonUnitalNormedRing γ
    inst✝ : Nontrivial γ
    h : Eq (compactlySupported α γ) Top.top
    ⊢ IsCompact Set.univ
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : γ)
  simpa [tsupport, Function.support_const hx]
    using (mem_compactlySupported (f := const α x).mp (by simp [h])).isCompact


/-- A compactly supported continuous function is automatically bounded. This constructor gives
an object of `α →ᵇ γ` from `g : α → γ` and these assumptions. -/
def ofCompactSupport (g : α → γ) (hg₁ : Continuous g) (hg₂ : HasCompactSupport g) : α →ᵇ γ where
  toFun := g
  continuous_toFun := hg₁
  map_bounded' := by
    /-
      α : Type u_1
      γ : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : NonUnitalNormedRing γ
      g : α → γ
      hg₁ : Continuous g
      hg₂ : HasCompactSupport g
      ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := g, continuous_toFu …
    -/
    obtain (hs | hs) := (tsupport g).eq_empty_or_nonempty
      /-
        case inl
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : α → γ
        hg₁ : Continuous g
        hg₂ : HasCompactSupport g
        hs : Eq (tsupport g) EmptyCollection.emptyCollection
        ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := g, continuous_toFu …
      -/
    · exact ⟨0, by simp [tsupport_eq_empty_iff.mp hs]⟩
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : α → γ
        hg₁ : Continuous g
        hg₂ : HasCompactSupport g
        hs : (tsupport g).Nonempty
        ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := g, continuous_toFu …
      -/
    · obtain ⟨z, _, hmax⟩ := hg₂.exists_isMaxOn hs <| hg₁.norm.continuousOn
      /-
        case inr.intro.intro
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : α → γ
        hg₁ : Continuous g
        hg₂ : HasCompactSupport g
        hs : (tsupport g).Nonempty
        z : α
        left✝ : Membership.mem (tsupport g) z
        hmax : IsMaxOn (fun x => Norm.norm (g x)) (tsupport g) z
        ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := g, continuous_toFu …
      -/
      refine ⟨2 * ‖g z‖, dist_le_two_norm' fun x ↦ ?_⟩
      /-
        case inr.intro.intro
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : α → γ
        hg₁ : Continuous g
        hg₂ : HasCompactSupport g
        hs : (tsupport g).Nonempty
        z : α
        left✝ : Membership.mem (tsupport g) z
        hmax : IsMaxOn (fun x => Norm.norm (g x)) (tsupport g) z
        x : α
        ⊢ LE.le (Norm.norm ({ toFun := g, continuous_toFun := hg₁ }.toFun x)) (Norm.no …
      -/
      by_cases hx : x ∈ tsupport g
        /-
          case pos
          α : Type u_1
          γ : Type u_2
          inst✝¹ : TopologicalSpace α
          inst✝ : NonUnitalNormedRing γ
          g : α → γ
          hg₁ : Continuous g
          hg₂ : HasCompactSupport g
          hs : (tsupport g).Nonempty
          z : α
          left✝ : Membership.mem (tsupport g) z
          hmax : IsMaxOn (fun x => Norm.norm (g x)) (tsupport g) z
          x : α
          hx : Membership.mem (tsupport g) x
          ⊢ LE.le (Norm.norm ({ toFun := g, continuous_toFun := hg₁ }.toFun x)) (Norm.no …
        -/
      · exact isMaxOn_iff.mp hmax x hx
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          γ : Type u_2
          inst✝¹ : TopologicalSpace α
          inst✝ : NonUnitalNormedRing γ
          g : α → γ
          hg₁ : Continuous g
          hg₂ : HasCompactSupport g
          hs : (tsupport g).Nonempty
          z : α
          left✝ : Membership.mem (tsupport g) z
          hmax : IsMaxOn (fun x => Norm.norm (g x)) (tsupport g) z
          x : α
          hx : Not (Membership.mem (tsupport g) x)
          ⊢ LE.le (Norm.norm ({ toFun := g, continuous_toFun := hg₁ }.toFun x)) (Norm.no …
        -/
      · simp [image_eq_zero_of_nmem_tsupport hx]
        /-
          🎉 no goals
        -/


lemma ofCompactSupport_mem (g : α → γ) (hg₁ : Continuous g) (hg₂ : HasCompactSupport g) :
    ofCompactSupport g hg₁ hg₂ ∈ C_cb(α, γ) := mem_compactlySupported.mpr hg₂


instance : SMul C(α, γ) C_cb(α, γ) where
  smul := fun (g : C(α, γ)) => (fun (f : C_cb(α, γ)) =>
    ⟨ofCompactSupport (g * (f : α →ᵇ γ) : α → γ) (Continuous.mul g.2 f.1.1.2)
    (HasCompactSupport.mul_left (mem_compactlySupported.mp f.2)), by
      /-
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : ContinuousMap α γ
        f : Subtype fun x => Membership.mem (compactlySupported α γ) x
        ⊢ Membership.mem (compactlySupported α γ) (ofCompactSupport (HMul.hMul ⇑g ⇑↑f) …
      -/
      apply mem_compactlySupported.mpr
      /-
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : ContinuousMap α γ
        f : Subtype fun x => Membership.mem (compactlySupported α γ) x
        ⊢ HasCompactSupport ⇑(ofCompactSupport (HMul.hMul ⇑g ⇑↑f) ⋯ ⋯)
      -/
      rw [ofCompactSupport]
      /-
        α : Type u_1
        γ : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : NonUnitalNormedRing γ
        g : ContinuousMap α γ
        f : Subtype fun x => Membership.mem (compactlySupported α γ) x
        ⊢ HasCompactSupport ⇑{ toFun := HMul.hMul ⇑g ⇑↑f, continuous_toFun := ⋯, map_b …
      -/
      exact HasCompactSupport.mul_left <| mem_compactlySupported.mp f.2
      /-
        🎉 no goals
      -/
    ⟩)


