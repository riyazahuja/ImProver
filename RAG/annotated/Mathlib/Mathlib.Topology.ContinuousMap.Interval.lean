/-- The embedding into an interval from a sub-interval lying on the left, as a `ContinuousMap`. -/
def IccInclusionLeft : C(Icc a b, Icc a c) :=
  .inclusion <| Icc_subset_Icc le_rfl Fact.out


/-- The embedding into an interval from a sub-interval lying on the right, as a `ContinuousMap`. -/
def IccInclusionRight : C(Icc b c, Icc a c) :=
  .inclusion <| Icc_subset_Icc Fact.out le_rfl


/-- The map `projIcc` from `α` onto an interval in `α`, as a `ContinuousMap`. -/
def projIccCM : C(α, Icc a b) :=
  ⟨projIcc a b Fact.out, continuous_projIcc⟩


/-- The extension operation from continuous maps on an interval to continuous maps on the whole
  type, as a `ContinuousMap`. -/
def IccExtendCM : C(C(Icc a b, E), C(α, E)) where
  toFun f := f.comp projIccCM
  continuous_toFun := continuous_precomp projIccCM


@[simp]
theorem IccExtendCM_of_mem {f : C(Icc a b, E)} {x : α} (hx : x ∈ Icc a b) :
    IccExtendCM f x = f ⟨x, hx⟩ := by
  /-
    α : Type u_1
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderTopology α
    a b : α
    inst✝¹ : Fact (LE.le a b)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    x : α
    hx : Membership.mem (Set.Icc a b) x
    ⊢ Eq ((ContinuousMap.IccExtendCM f) x) (f ⟨x, hx⟩)
  -/
  simp [IccExtendCM, projIccCM, projIcc, hx.1, hx.2]
  /-
    🎉 no goals
  -/


/-- The concatenation of two continuous maps defined on adjacent intervals. If the values of the
functions on the common bound do not agree, this is defined as an arbitrarily chosen constant
map. See `concatCM` for the corresponding map on the subtype of compatible function pairs. -/
noncomputable def concat (f : C(Icc a b, E)) (g : C(Icc b c, E)) :
    C(Icc a c, E) := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ⊢ ContinuousMap (↑(Set.Icc a c)) E
  -/
  by_cases hb : f ⊤ = g ⊥
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      ⊢ ContinuousMap (↑(Set.Icc a c)) E
    -/
  · let h (t : α) : E := if t ≤ b then IccExtendCM f t else IccExtendCM g t
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      h : α → E := fun t => ite (LE.le t b) ((ContinuousMap.IccExtendCM f) t) ((Cont …
      ⊢ ContinuousMap (↑(Set.Icc a c)) E
    -/
    suffices Continuous h from ⟨fun t => h t, by fun_prop⟩
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      h : α → E := fun t => ite (LE.le t b) ((ContinuousMap.IccExtendCM f) t) ((Cont …
      ⊢ Continuous h
    -/
    apply Continuous.if_le (by fun_prop) (by fun_prop) continuous_id continuous_const
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      h : α → E := fun t => ite (LE.le t b) ((ContinuousMap.IccExtendCM f) t) ((Cont …
      ⊢ ∀ (x : α), Eq (id x) b → Eq ((ContinuousMap.IccExtendCM f) x) ((ContinuousMa …
    -/
    rintro x rfl
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a c : α
      E : Type u_2
      inst✝² : TopologicalSpace E
      x : α
      inst✝¹ : Fact (LE.le a (id x))
      inst✝ : Fact (LE.le (id x) c)
      f : ContinuousMap (↑(Set.Icc a (id x))) E
      g : ContinuousMap (↑(Set.Icc (id x) c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      h : α → E := fun t => ite (LE.le t (id x)) ((ContinuousMap.IccExtendCM f) t) ( …
      ⊢ Eq ((ContinuousMap.IccExtendCM f) x) ((ContinuousMap.IccExtendCM g) x)
    -/
    simpa [IccExtendCM, projIccCM]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Not (Eq (f Top.top) (g Bot.bot))
      ⊢ ContinuousMap (↑(Set.Icc a c)) E
    -/
  · exact .const _ (f ⊥) -- junk value
    /-
      🎉 no goals
    -/


theorem concat_comp_IccInclusionLeft (hb : f ⊤ = g ⊥) :
    (concat f g).comp IccInclusionLeft = f := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    ⊢ Eq ((f.concat g).comp ContinuousMap.IccInclusionLeft) f
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    x : ↑(Set.Icc a b)
    ⊢ Eq (((f.concat g).comp ContinuousMap.IccInclusionLeft) x) (f x)
  -/
  simp [concat, IccExtendCM, hb, IccInclusionLeft, projIccCM, inclusion, x.2.2]
  /-
    🎉 no goals
  -/


theorem concat_comp_IccInclusionRight (hb : f ⊤ = g ⊥) :
    (concat f g).comp IccInclusionRight = g := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    ⊢ Eq ((f.concat g).comp ContinuousMap.IccInclusionRight) g
  -/
  ext ⟨x, hx⟩
  /-
    case h.mk
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    x : α
    hx : Membership.mem (Set.Icc b c) x
    ⊢ Eq (((f.concat g).comp ContinuousMap.IccInclusionRight) ⟨x, hx⟩) (g ⟨x, hx⟩)
  -/
  obtain rfl | hxb := eq_or_ne x b
    /-
      case h.mk.inl
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a c : α
      E : Type u_2
      inst✝² : TopologicalSpace E
      x : α
      inst✝¹ : Fact (LE.le a x)
      inst✝ : Fact (LE.le x c)
      f : ContinuousMap (↑(Set.Icc a x)) E
      g : ContinuousMap (↑(Set.Icc x c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      hx : Membership.mem (Set.Icc x c) x
      ⊢ Eq (((f.concat g).comp ContinuousMap.IccInclusionRight) ⟨x, hx⟩) (g ⟨x, hx⟩)
    -/
  · simpa [concat, IccInclusionRight, IccExtendCM, projIccCM, inclusion, hb]
    /-
      🎉 no goals
    -/
    /-
      case h.mk.inr
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      x : α
      hx : Membership.mem (Set.Icc b c) x
      hxb : Ne x b
      ⊢ Eq (((f.concat g).comp ContinuousMap.IccInclusionRight) ⟨x, hx⟩) (g ⟨x, hx⟩)
    -/
  · have h : ¬ x ≤ b := lt_of_le_of_ne hx.1 (Ne.symm hxb) |>.not_le
    /-
      case h.mk.inr
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      hb : Eq (f Top.top) (g Bot.bot)
      x : α
      hx : Membership.mem (Set.Icc b c) x
      hxb : Ne x b
      h : Not (LE.le x b)
      ⊢ Eq (((f.concat g).comp ContinuousMap.IccInclusionRight) ⟨x, hx⟩) (g ⟨x, hx⟩)
    -/
    simp [concat, hb, IccInclusionRight, h, IccExtendCM, projIccCM, projIcc, inclusion, hx.2, hx.1]
    /-
      🎉 no goals
    -/


@[simp]
theorem concat_left (hb : f ⊤ = g ⊥) {t : Icc a c} (ht : t ≤ b) :
    concat f g t = f ⟨t, t.2.1, ht⟩ := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    t : ↑(Set.Icc a c)
    ht : LE.le (↑t) b
    ⊢ Eq ((f.concat g) t) (f ⟨↑t, ⋯⟩)
  -/
  nth_rewrite 2 [← concat_comp_IccInclusionLeft hb]
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    t : ↑(Set.Icc a c)
    ht : LE.le (↑t) b
    ⊢ Eq ((f.concat g) t) (((f.concat g).comp ContinuousMap.IccInclusionLeft) ⟨↑t, …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem concat_right (hb : f ⊤ = g ⊥) {t : Icc a c} (ht : b ≤ t) :
    concat f g t = g ⟨t, ht, t.2.2⟩ := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    t : ↑(Set.Icc a c)
    ht : LE.le b ↑t
    ⊢ Eq ((f.concat g) t) (g ⟨↑t, ⋯⟩)
  -/
  nth_rewrite 2 [← concat_comp_IccInclusionRight hb]
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    hb : Eq (f Top.top) (g Bot.bot)
    t : ↑(Set.Icc a c)
    ht : LE.le b ↑t
    ⊢ Eq ((f.concat g) t) (((f.concat g).comp ContinuousMap.IccInclusionRight) ⟨↑t …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tendsto_concat {ι : Type*} {p : Filter ι} {F : ι → C(Icc a b, E)} {G : ι → C(Icc b c, E)}
    (hfg : ∀ᶠ i in p, (F i) ⊤ = (G i) ⊥) (hfg' : f ⊤ = g ⊥)
    (hf : Tendsto F p (𝓝 f)) (hg : Tendsto G p (𝓝 g)) :
    Tendsto (fun i => concat (F i) (G i)) p (𝓝 (concat f g)) := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf : Filter.Tendsto F p (nhds f)
    hg : Filter.Tendsto G p (nhds g)
    ⊢ Filter.Tendsto (fun i => (F i).concat (G i)) p (nhds (f.concat g))
  -/
  rw [tendsto_nhds_compactOpen] at hf hg ⊢
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    hg : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    ⊢ ∀ (K : Set ↑(Set.Icc a c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.Maps …
  -/
  rintro K hK U hU hfgU
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    hg : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    K : Set ↑(Set.Icc a c)
    hK : IsCompact K
    U : Set E
    hU : IsOpen U
    hfgU : Set.MapsTo (⇑(f.concat g)) K U
    ⊢ Filter.Eventually (fun a_1 => Set.MapsTo (⇑((F a_1).concat (G a_1))) K U) p
  -/
  have h : b ∈ Icc a c := ⟨Fact.out, Fact.out⟩
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    hg : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    K : Set ↑(Set.Icc a c)
    hK : IsCompact K
    U : Set E
    hU : IsOpen U
    hfgU : Set.MapsTo (⇑(f.concat g)) K U
    h : Membership.mem (Set.Icc a c) b
    ⊢ Filter.Eventually (fun a_1 => Set.MapsTo (⇑((F a_1).concat (G a_1))) K U) p
  -/
  let K₁ : Set (Icc a b) := projIccCM '' (Subtype.val '' (K ∩ Iic ⟨b, h⟩))
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    hg : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    K : Set ↑(Set.Icc a c)
    hK : IsCompact K
    U : Set E
    hU : IsOpen U
    hfgU : Set.MapsTo (⇑(f.concat g)) K U
    h : Membership.mem (Set.Icc a c) b
    K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
    ⊢ Filter.Eventually (fun a_1 => Set.MapsTo (⇑((F a_1).concat (G a_1))) K U) p
  -/
  let K₂ : Set (Icc b c) := projIccCM '' (Subtype.val '' (K ∩ Ici ⟨b, h⟩))
  have hK₁ : IsCompact K₁ :=
    hK.inter_right isClosed_Iic |>.image continuous_subtype_val |>.image projIccCM.continuous
  have hK₂ : IsCompact K₂ :=
    hK.inter_right isClosed_Ici |>.image continuous_subtype_val |>.image projIccCM.continuous
  have hfU : MapsTo f K₁ U := by
    rw [← concat_comp_IccInclusionLeft hfg']
    apply hfgU.comp
    rintro x ⟨y, ⟨⟨z, hz⟩, ⟨h1, (h2 : z ≤ b)⟩, rfl⟩, rfl⟩
    simpa [projIccCM, projIcc, h2, hz.1] using h1
  have hgU : MapsTo g K₂ U := by
    rw [← concat_comp_IccInclusionRight hfg']
    apply hfgU.comp
    rintro x ⟨y, ⟨⟨z, hz⟩, ⟨h1, (h2 : b ≤ z)⟩, rfl⟩, rfl⟩
    simpa [projIccCM, projIcc, h2, hz.2] using h1
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    hg : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set.M …
    K : Set ↑(Set.Icc a c)
    hK : IsCompact K
    U : Set E
    hU : IsOpen U
    hfgU : Set.MapsTo (⇑(f.concat g)) K U
    h : Membership.mem (Set.Icc a c) b
    K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
    K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
    hK₁ : IsCompact K₁
    hK₂ : IsCompact K₂
    hfU : Set.MapsTo (⇑f) K₁ U
    hgU : Set.MapsTo (⇑g) K₂ U
    ⊢ Filter.Eventually (fun a_1 => Set.MapsTo (⇑((F a_1).concat (G a_1))) K U) p
  -/
  filter_upwards [hf K₁ hK₁ U hU hfU, hg K₂ hK₂ U hU hgU, hfg] with i hf hg hfg x hx
  /-
    case h
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    f : ContinuousMap (↑(Set.Icc a b)) E
    g : ContinuousMap (↑(Set.Icc b c)) E
    ι : Type u_3
    p : Filter ι
    F : ι → ContinuousMap (↑(Set.Icc a b)) E
    G : ι → ContinuousMap (↑(Set.Icc b c)) E
    hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
    hfg' : Eq (f Top.top) (g Bot.bot)
    hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
    hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
    K : Set ↑(Set.Icc a c)
    hK : IsCompact K
    U : Set E
    hU : IsOpen U
    hfgU : Set.MapsTo (⇑(f.concat g)) K U
    h : Membership.mem (Set.Icc a c) b
    K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
    K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
    hK₁ : IsCompact K₁
    hK₂ : IsCompact K₂
    hfU : Set.MapsTo (⇑f) K₁ U
    hgU : Set.MapsTo (⇑g) K₂ U
    i : ι
    hf : Set.MapsTo (⇑(F i)) K₁ U
    hg : Set.MapsTo (⇑(G i)) K₂ U
    hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
    x : ↑(Set.Icc a c)
    hx : Membership.mem K x
    ⊢ Membership.mem U (((F i).concat (G i)) x)
  -/
  by_cases hxb : x ≤ b
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : LE.le (↑x) b
      ⊢ Membership.mem U (((F i).concat (G i)) x)
    -/
  · rw [concat_left hfg hxb]
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : LE.le (↑x) b
      ⊢ Membership.mem U ((F i) ⟨↑x, ⋯⟩)
    -/
    refine hf ⟨x, ⟨x, ⟨hx, hxb⟩, rfl⟩, ?_⟩
    /-
      case pos
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : LE.le (↑x) b
      ⊢ Eq (ContinuousMap.projIccCM ↑x) ⟨↑x, ⋯⟩
    -/
    simp [projIccCM, projIcc, hxb, x.2.1]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : Not (LE.le (↑x) b)
      ⊢ Membership.mem U (((F i).concat (G i)) x)
    -/
  · replace hxb : b ≤ x := lt_of_not_le hxb |>.le
    /-
      case neg
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : LE.le b ↑x
      ⊢ Membership.mem U (((F i).concat (G i)) x)
    -/
    rw [concat_right hfg hxb]
    /-
      case neg
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : LE.le b ↑x
      ⊢ Membership.mem U ((G i) ⟨↑x, ⋯⟩)
    -/
    refine hg ⟨x, ⟨x, ⟨hx, hxb⟩, rfl⟩, ?_⟩
    /-
      case neg
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ι : Type u_3
      p : Filter ι
      F : ι → ContinuousMap (↑(Set.Icc a b)) E
      G : ι → ContinuousMap (↑(Set.Icc b c)) E
      hfg✝ : Filter.Eventually (fun i => Eq ((F i) Top.top) ((G i) Bot.bot)) p
      hfg' : Eq (f Top.top) (g Bot.bot)
      hf✝ : ∀ (K : Set ↑(Set.Icc a b)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      hg✝ : ∀ (K : Set ↑(Set.Icc b c)), IsCompact K → ∀ (U : Set E), IsOpen U → Set. …
      K : Set ↑(Set.Icc a c)
      hK : IsCompact K
      U : Set E
      hU : IsOpen U
      hfgU : Set.MapsTo (⇑(f.concat g)) K U
      h : Membership.mem (Set.Icc a c) b
      K₁ : Set ↑(Set.Icc a b) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      K₂ : Set ↑(Set.Icc b c) := Set.image (⇑ContinuousMap.projIccCM) (Set.image Sub …
      hK₁ : IsCompact K₁
      hK₂ : IsCompact K₂
      hfU : Set.MapsTo (⇑f) K₁ U
      hgU : Set.MapsTo (⇑g) K₂ U
      i : ι
      hf : Set.MapsTo (⇑(F i)) K₁ U
      hg : Set.MapsTo (⇑(G i)) K₂ U
      hfg : Eq ((F i) Top.top) ((G i) Bot.bot)
      x : ↑(Set.Icc a c)
      hx : Membership.mem K x
      hxb : LE.le b ↑x
      ⊢ Eq (ContinuousMap.projIccCM ↑x) ⟨↑x, ⋯⟩
    -/
    simp [projIccCM, projIcc, hxb, x.2.2]
    /-
      🎉 no goals
    -/


/-- The concatenation of compatible pairs of continuous maps on adjacent intervals, defined as a
`ContinuousMap` on a subtype of the product. -/
noncomputable def concatCM :
    C({fg : C(Icc a b, E) × C(Icc b c, E) // fg.1 ⊤ = fg.2 ⊥}, C(Icc a c, E))
    where
  toFun fg := concat fg.val.1 fg.val.2
  continuous_toFun := by
    /-
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      ⊢ Continuous fun fg => (↑fg).1.concat (↑fg).2
    -/
    let S : Set (C(Icc a b, E) × C(Icc b c, E)) := {fg | fg.1 ⊤ = fg.2 ⊥}
    /-
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      S : Set (Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c …
      ⊢ Continuous fun fg => (↑fg).1.concat (↑fg).2
    -/
    change Continuous (S.restrict concat.uncurry)
    /-
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      S : Set (Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c …
      ⊢ Continuous (S.restrict (Function.uncurry ContinuousMap.concat))
    -/
    refine continuousOn_iff_continuous_restrict.mp (fun fg hfg => ?_)
    /-
      α : Type u_1
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      a b c : α
      inst✝² : Fact (LE.le a b)
      inst✝¹ : Fact (LE.le b c)
      E : Type u_2
      inst✝ : TopologicalSpace E
      f : ContinuousMap (↑(Set.Icc a b)) E
      g : ContinuousMap (↑(Set.Icc b c)) E
      S : Set (Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c …
      fg : Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c)) E)
      hfg : Membership.mem S fg
      ⊢ ContinuousWithinAt (Function.uncurry ContinuousMap.concat) S fg
    -/
    refine tendsto_concat ?_ hfg ?_ ?_
      /-
        case refine_1
        α : Type u_1
        inst✝⁵ : LinearOrder α
        inst✝⁴ : TopologicalSpace α
        inst✝³ : OrderTopology α
        a b c : α
        inst✝² : Fact (LE.le a b)
        inst✝¹ : Fact (LE.le b c)
        E : Type u_2
        inst✝ : TopologicalSpace E
        f : ContinuousMap (↑(Set.Icc a b)) E
        g : ContinuousMap (↑(Set.Icc b c)) E
        S : Set (Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c …
        fg : Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c)) E)
        hfg : Membership.mem S fg
        ⊢ Filter.Eventually (fun i => Eq (i.1 Top.top) (i.2 Bot.bot)) (nhdsWithin fg S)
      -/
    · exact eventually_nhdsWithin_of_forall (fun _ => id)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        inst✝⁵ : LinearOrder α
        inst✝⁴ : TopologicalSpace α
        inst✝³ : OrderTopology α
        a b c : α
        inst✝² : Fact (LE.le a b)
        inst✝¹ : Fact (LE.le b c)
        E : Type u_2
        inst✝ : TopologicalSpace E
        f : ContinuousMap (↑(Set.Icc a b)) E
        g : ContinuousMap (↑(Set.Icc b c)) E
        S : Set (Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c …
        fg : Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c)) E)
        hfg : Membership.mem S fg
        ⊢ Filter.Tendsto Prod.fst (nhdsWithin fg S) (nhds fg.1)
      -/
    · exact tendsto_nhdsWithin_of_tendsto_nhds continuousAt_fst
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        α : Type u_1
        inst✝⁵ : LinearOrder α
        inst✝⁴ : TopologicalSpace α
        inst✝³ : OrderTopology α
        a b c : α
        inst✝² : Fact (LE.le a b)
        inst✝¹ : Fact (LE.le b c)
        E : Type u_2
        inst✝ : TopologicalSpace E
        f : ContinuousMap (↑(Set.Icc a b)) E
        g : ContinuousMap (↑(Set.Icc b c)) E
        S : Set (Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c …
        fg : Prod (ContinuousMap (↑(Set.Icc a b)) E) (ContinuousMap (↑(Set.Icc b c)) E)
        hfg : Membership.mem S fg
        ⊢ Filter.Tendsto Prod.snd (nhdsWithin fg S) (nhds fg.2)
      -/
    · exact tendsto_nhdsWithin_of_tendsto_nhds continuousAt_snd
      /-
        🎉 no goals
      -/


@[simp]
theorem concatCM_left {x : Icc a c} (hx : x ≤ b)
    {fg : {fg : C(Icc a b, E) × C(Icc b c, E) // fg.1 ⊤ = fg.2 ⊥}} :
    concatCM fg x = fg.1.1 ⟨x.1, x.2.1, hx⟩ := by
  /-
    α : Type u_1
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    a b c : α
    inst✝² : Fact (LE.le a b)
    inst✝¹ : Fact (LE.le b c)
    E : Type u_2
    inst✝ : TopologicalSpace E
    x : ↑(Set.Icc a c)
    hx : LE.le (↑x) b
    fg : Subtype fun fg => Eq (fg.1 Top.top) (fg.2 Bot.bot)
    ⊢ Eq ((ContinuousMap.concatCM fg) x) ((↑fg).1 ⟨↑x, ⋯⟩)
  -/
  exact concat_left fg.2 hx
  /-
    🎉 no goals
  -/


@[simp]
theorem concatCM_right {x : Icc a c} (hx : b ≤ x)
    {fg : {fg : C(Icc a b, E) × C(Icc b c, E) // fg.1 ⊤ = fg.2 ⊥}} :
    concatCM fg x = fg.1.2 ⟨x.1, hx, x.2.2⟩ :=
  concat_right fg.2 hx


