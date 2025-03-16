/-- `Filter α` is an atomic type: for every filter there exists an ultrafilter that is less than or
equal to this filter. -/
instance : IsAtomic (Filter α) :=
  IsAtomic.of_isChain_bounded fun c hc hne hb =>
    ⟨sInf c, (sInf_neBot_of_directed' hne (show IsChain (· ≥ ·) c from hc.symm).directedOn hb).ne,
      fun _ hx => sInf_le hx⟩


/-- An ultrafilter is a minimal (maximal in the set order) proper filter. -/
structure Ultrafilter (α : Type*) extends Filter α where
  /-- An ultrafilter is nontrivial. -/
  protected neBot' : NeBot toFilter
  /-- If `g` is a nontrivial filter that is less than or equal to an ultrafilter, then it is greater
  than or equal to the ultrafilter. -/
  protected le_of_le : ∀ g, Filter.NeBot g → g ≤ toFilter → toFilter ≤ g


instance : CoeTC (Ultrafilter α) (Filter α) :=
  ⟨Ultrafilter.toFilter⟩


instance : Membership (Set α) (Ultrafilter α) :=
  ⟨fun f s => s ∈ (f : Filter α)⟩


theorem unique (f : Ultrafilter α) {g : Filter α} (h : g ≤ f) (hne : NeBot g := by infer_instance) :
    g = f :=
  le_antisymm h <| f.le_of_le g hne h


instance neBot (f : Ultrafilter α) : NeBot (f : Filter α) :=
  f.neBot'


protected theorem isAtom (f : Ultrafilter α) : IsAtom (f : Filter α) :=
  ⟨f.neBot.ne, fun _ hgf => by_contra fun hg => hgf.ne <| f.unique hgf.le ⟨hg⟩⟩


@[simp, norm_cast]
theorem mem_coe : s ∈ (f : Filter α) ↔ s ∈ f :=
  Iff.rfl


theorem coe_injective : Injective ((↑) : Ultrafilter α → Filter α)
                                    /-
                                      α : Type u
                                      f : Filter α
                                      h₁ : f.NeBot
                                      h₂ : ∀ (g : Filter α), g.NeBot → LE.le g f → LE.le f g
                                      g : Filter α
                                      neBot'✝ : g.NeBot
                                      le_of_le✝ : ∀ (g_1 : Filter α), g_1.NeBot → LE.le g_1 g → LE.le g g_1
                                      x✝ : Eq ↑{ toFilter := f, neBot' := h₁, le_of_le := h₂ } ↑{ toFilter := g, neB …
                                      ⊢ Eq { toFilter := f, neBot' := h₁, le_of_le := h₂ } { toFilter := g, neBot' : …
                                    -/
  | ⟨f, h₁, h₂⟩, ⟨g, _, _⟩, _ => by congr
                                    /-
                                      🎉 no goals
                                    -/


theorem eq_of_le {f g : Ultrafilter α} (h : (f : Filter α) ≤ g) : f = g :=
                 /-
                   α : Type u
                   f g : Ultrafilter α
                   h : LE.le ↑f ↑g
                   ⊢ (↑f).NeBot
                 -/
  coe_injective (g.unique h)
                 /-
                   🎉 no goals
                 -/


@[simp, norm_cast]
theorem coe_le_coe {f g : Ultrafilter α} : (f : Filter α) ≤ g ↔ f = g :=
  ⟨fun h => eq_of_le h, fun h => h ▸ le_rfl⟩


@[simp, norm_cast]
theorem coe_inj : (f : Filter α) = g ↔ f = g :=
  coe_injective.eq_iff


@[ext]
theorem ext ⦃f g : Ultrafilter α⦄ (h : ∀ s, s ∈ f ↔ s ∈ g) : f = g :=
  coe_injective <| Filter.ext h


theorem le_of_inf_neBot (f : Ultrafilter α) {g : Filter α} (hg : NeBot (↑f ⊓ g)) : ↑f ≤ g :=
  le_of_inf_eq (f.unique inf_le_left hg)


theorem le_of_inf_neBot' (f : Ultrafilter α) {g : Filter α} (hg : NeBot (g ⊓ f)) : ↑f ≤ g :=
                          /-
                            α : Type u
                            f : Ultrafilter α
                            g : Filter α
                            hg : (Min.min g ↑f).NeBot
                            ⊢ (Min.min (↑f) g).NeBot
                          -/
  f.le_of_inf_neBot <| by rwa [inf_comm]
                          /-
                            🎉 no goals
                          -/


theorem inf_neBot_iff {f : Ultrafilter α} {g : Filter α} : NeBot (↑f ⊓ g) ↔ ↑f ≤ g :=
  ⟨le_of_inf_neBot f, fun h => (inf_of_le_left h).symm ▸ f.neBot⟩


theorem disjoint_iff_not_le {f : Ultrafilter α} {g : Filter α} : Disjoint (↑f) g ↔ ¬↑f ≤ g := by
  /-
    α : Type u
    f : Ultrafilter α
    g : Filter α
    ⊢ Iff (Disjoint (↑f) g) (Not (LE.le (↑f) g))
  -/
  rw [← inf_neBot_iff, neBot_iff, Ne, not_not, disjoint_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_not_mem_iff : sᶜ ∉ f ↔ s ∈ f :=
  ⟨fun hsc =>
    le_principal_iff.1 <|
                                                             /-
                                                               α : Type u
                                                               f : Ultrafilter α
                                                               s : Set α
                                                               hsc : Not (Membership.mem f (HasCompl.compl s))
                                                               h : Eq (Min.min (↑f) (Filter.principal s)) Bot.bot
                                                               ⊢ Eq (Min.min (↑f) (Filter.principal (HasCompl.compl (HasCompl.compl s)))) Bot …
                                                             -/
      f.le_of_inf_neBot ⟨fun h => hsc <| mem_of_eq_bot <| by rwa [compl_compl]⟩,
                                                             /-
                                                               🎉 no goals
                                                             -/
    compl_not_mem⟩


@[simp]
theorem frequently_iff_eventually : (∃ᶠ x in f, p x) ↔ ∀ᶠ x in f, p x :=
  compl_not_mem_iff


alias ⟨_root_.Filter.Frequently.eventually, _⟩ := frequently_iff_eventually


                                                     /-
                                                       α : Type u
                                                       f : Ultrafilter α
                                                       s : Set α
                                                       ⊢ Iff (Membership.mem f (HasCompl.compl s)) (Not (Membership.mem f s))
                                                     -/
theorem compl_mem_iff_not_mem : sᶜ ∈ f ↔ s ∉ f := by rw [← compl_not_mem_iff, compl_compl]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem diff_mem_iff (f : Ultrafilter α) : s \ t ∈ f ↔ s ∈ f ∧ t ∉ f :=
  inter_mem_iff.trans <| and_congr Iff.rfl compl_mem_iff_not_mem


/-- If `sᶜ ∉ f ↔ s ∈ f`, then `f` is an ultrafilter. The other implication is given by
`Ultrafilter.compl_not_mem_iff`. -/
def ofComplNotMemIff (f : Filter α) (h : ∀ s, sᶜ ∉ f ↔ s ∈ f) : Ultrafilter α where
  toFilter := f
                          /-
                            α : Type u
                            β : Type v
                            γ : Type u_1
                            f✝ g : Ultrafilter α
                            s t : Set α
                            p q : α → Prop
                            f : Filter α
                            h : ∀ (s : Set α), Iff (Not (Membership.mem f (HasCompl.compl s))) (Membership …
                            hf : Eq f Bot.bot
                            ⊢ False
                          -/
  neBot' := ⟨fun hf => by simp [hf] at h⟩
                          /-
                            🎉 no goals
                          -/
  le_of_le _ _ hgf s hs := (h s).1 fun hsc => compl_not_mem hs (hgf hsc)


/-- If `f : Filter α` is an atom, then it is an ultrafilter. -/
def ofAtom (f : Filter α) (hf : IsAtom f) : Ultrafilter α where
  toFilter := f
  neBot' := ⟨hf.1⟩
  le_of_le g hg := (isAtom_iff_le_of_ge.1 hf).2 g hg.ne


theorem nonempty_of_mem (hs : s ∈ f) : s.Nonempty :=
  Filter.nonempty_of_mem hs


theorem ne_empty_of_mem (hs : s ∈ f) : s ≠ ∅ :=
  (nonempty_of_mem hs).ne_empty


@[simp]
theorem empty_not_mem : ∅ ∉ f :=
  Filter.empty_not_mem (f : Filter α)


@[simp]
theorem le_sup_iff {u : Ultrafilter α} {f g : Filter α} : ↑u ≤ f ⊔ g ↔ ↑u ≤ f ∨ ↑u ≤ g :=
                      /-
                        α : Type u
                        u : Ultrafilter α
                        f g : Filter α
                        ⊢ Iff (Not (LE.le (↑u) (Max.max f g))) (Not (Or (LE.le (↑u) f) (LE.le (↑u) g)))
                      -/
  not_iff_not.1 <| by simp only [← disjoint_iff_not_le, not_or, disjoint_sup_right]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem union_mem_iff : s ∪ t ∈ f ↔ s ∈ f ∨ t ∈ f := by
  /-
    α : Type u
    f : Ultrafilter α
    s t : Set α
    ⊢ Iff (Membership.mem f (Union.union s t)) (Or (Membership.mem f s) (Membershi …
  -/
  simp only [← mem_coe, ← le_principal_iff, ← sup_principal, le_sup_iff]
  /-
    🎉 no goals
  -/


theorem mem_or_compl_mem (f : Ultrafilter α) (s : Set α) : s ∈ f ∨ sᶜ ∈ f :=
  or_iff_not_imp_left.2 compl_mem_iff_not_mem.2


protected theorem em (f : Ultrafilter α) (p : α → Prop) : (∀ᶠ x in f, p x) ∨ ∀ᶠ x in f, ¬p x :=
  f.mem_or_compl_mem { x | p x }


theorem eventually_or : (∀ᶠ x in f, p x ∨ q x) ↔ (∀ᶠ x in f, p x) ∨ ∀ᶠ x in f, q x :=
  union_mem_iff


theorem eventually_not : (∀ᶠ x in f, ¬p x) ↔ ¬∀ᶠ x in f, p x :=
  compl_mem_iff_not_mem


theorem eventually_imp : (∀ᶠ x in f, p x → q x) ↔ (∀ᶠ x in f, p x) → ∀ᶠ x in f, q x := by
  /-
    α : Type u
    f : Ultrafilter α
    p q : α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x → q x) ↑f) (Filter.Eventually (fun x => …
  -/
  simp only [imp_iff_not_or, eventually_or, eventually_not]
  /-
    🎉 no goals
  -/


theorem finite_sUnion_mem_iff {s : Set (Set α)} (hs : s.Finite) : ⋃₀ s ∈ f ↔ ∃ t ∈ s, t ∈ f :=
                             /-
                               α : Type u
                               f : Ultrafilter α
                               s : Set (Set α)
                               hs : s.Finite
                               ⊢ Iff (Membership.mem f EmptyCollection.emptyCollection.sUnion) (Exists fun t  …
                             -/
  Finite.induction_on hs (by simp) fun _ _ his => by
                             /-
                               🎉 no goals
                             -/
    /-
      α : Type u
      f : Ultrafilter α
      s : Set (Set α)
      hs : s.Finite
      a✝ : Set α
      s✝ : Set (Set α)
      x✝¹ : Not (Membership.mem s✝ a✝)
      x✝ : s✝.Finite
      his : Iff (Membership.mem f s✝.sUnion) (Exists fun t => And (Membership.mem s✝ …
      ⊢ Iff (Membership.mem f (Insert.insert a✝ s✝).sUnion) (Exists fun t => And (Me …
    -/
    simp [union_mem_iff, his, or_and_right, exists_or]
    /-
      🎉 no goals
    -/


theorem finite_biUnion_mem_iff {is : Set β} {s : β → Set α} (his : is.Finite) :
    (⋃ i ∈ is, s i) ∈ f ↔ ∃ i ∈ is, s i ∈ f := by
  /-
    α : Type u
    β : Type v
    f : Ultrafilter α
    is : Set β
    s : β → Set α
    his : is.Finite
    ⊢ Iff (Membership.mem f (Set.iUnion fun i => Set.iUnion fun h => s i)) (Exists …
  -/
  simp only [← sUnion_image, finite_sUnion_mem_iff (his.image s), exists_mem_image]
  /-
    🎉 no goals
  -/


lemma eventually_exists_mem_iff {is : Set β} {P : β → α → Prop} (his : is.Finite) :
    (∀ᶠ i in f, ∃ a ∈ is, P a i) ↔ ∃ a ∈ is, ∀ᶠ i in f, P a i := by
  /-
    α : Type u
    β : Type v
    f : Ultrafilter α
    is : Set β
    P : β → α → Prop
    his : is.Finite
    ⊢ Iff (Filter.Eventually (fun i => Exists fun a => And (Membership.mem is a) ( …
  -/
  simp only [Filter.Eventually, Ultrafilter.mem_coe]
  /-
    α : Type u
    β : Type v
    f : Ultrafilter α
    is : Set β
    P : β → α → Prop
    his : is.Finite
    ⊢ Iff (Membership.mem f (setOf fun x => Exists fun a => And (Membership.mem is …
  -/
  convert f.finite_biUnion_mem_iff his (s := P) with i
  /-
    case h.e'_1.h.e'_5
    α : Type u
    β : Type v
    f : Ultrafilter α
    is : Set β
    P : β → α → Prop
    his : is.Finite
    ⊢ Eq (setOf fun x => Exists fun a => And (Membership.mem is a) (P a x)) (Set.i …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma eventually_exists_iff [Finite β] {P : β → α → Prop} :
    (∀ᶠ i in f, ∃ a, P a i) ↔ ∃ a, ∀ᶠ i in f, P a i := by
  /-
    α : Type u
    β : Type v
    f : Ultrafilter α
    inst✝ : Finite β
    P : β → α → Prop
    ⊢ Iff (Filter.Eventually (fun i => Exists fun a => P a i) ↑f) (Exists fun a => …
  -/
  simpa using eventually_exists_mem_iff (f := f) (P := P) Set.finite_univ
  /-
    🎉 no goals
  -/


/-- Pushforward for ultrafilters. -/
nonrec def map (m : α → β) (f : Ultrafilter α) : Ultrafilter β :=
  ofComplNotMemIff (map m f) fun s => @compl_not_mem_iff _ f (m ⁻¹' s)


@[simp, norm_cast]
theorem coe_map (m : α → β) (f : Ultrafilter α) : (map m f : Filter β) = Filter.map m ↑f :=
  rfl


@[simp]
theorem mem_map {m : α → β} {f : Ultrafilter α} {s : Set β} : s ∈ map m f ↔ m ⁻¹' s ∈ f :=
  Iff.rfl


@[simp]
nonrec theorem map_id (f : Ultrafilter α) : f.map id = f :=
  coe_injective map_id


@[simp]
theorem map_id' (f : Ultrafilter α) : (f.map fun x => x) = f :=
  map_id _


@[simp]
nonrec theorem map_map (f : Ultrafilter α) (m : α → β) (n : β → γ) :
  (f.map m).map n = f.map (n ∘ m) :=
  coe_injective map_map


/-- The pullback of an ultrafilter along an injection whose range is large with respect to the given
ultrafilter. -/
nonrec def comap {m : α → β} (u : Ultrafilter β) (inj : Injective m) (large : Set.range m ∈ u) :
    Ultrafilter α where
  toFilter := comap m u
  neBot' := u.neBot'.comap_of_range_mem large
  le_of_le g hg hgu := by
    /-
      α : Type u
      β : Type v
      γ : Type u_1
      f g✝ : Ultrafilter α
      s t : Set α
      p q : α → Prop
      m : α → β
      u : Ultrafilter β
      inj : Function.Injective m
      large : Membership.mem u (Set.range m)
      g : Filter α
      hg : g.NeBot
      hgu : LE.le g (Filter.comap m ↑u)
      ⊢ LE.le (Filter.comap m ↑u) g
    -/
    simp only [← u.unique (map_le_iff_le_comap.2 hgu), comap_map inj, le_rfl]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_comap {m : α → β} (u : Ultrafilter β) (inj : Injective m) (large : Set.range m ∈ u)
    {s : Set α} : s ∈ u.comap inj large ↔ m '' s ∈ u :=
  mem_comap_iff inj large


@[simp, norm_cast]
theorem coe_comap {m : α → β} (u : Ultrafilter β) (inj : Injective m) (large : Set.range m ∈ u) :
    (u.comap inj large : Filter α) = Filter.comap m u :=
  rfl


@[simp]
nonrec theorem comap_id (f : Ultrafilter α) (h₀ : Injective (id : α → α) := injective_id)
                              /-
                                α : Type u
                                β : Type v
                                γ : Type u_1
                                f✝ g : Ultrafilter α
                                s t : Set α
                                p q : α → Prop
                                f : Ultrafilter α
                                h₀ : optParam (Function.Injective id) ⋯
                                ⊢ Membership.mem f (Set.range id)
                              -/
    (h₁ : range id ∈ f := (by rw [range_id]; exact univ_mem)) :
                                             /-
                                               🎉 no goals
                                             -/
    f.comap h₀ h₁ = f :=
  coe_injective comap_id


@[simp]
nonrec theorem comap_comap (f : Ultrafilter γ) {m : α → β} {n : β → γ} (inj₀ : Injective n)
    (large₀ : range n ∈ f) (inj₁ : Injective m) (large₁ : range m ∈ f.comap inj₀ large₀)
    (inj₂ : Injective (n ∘ m) := inj₀.comp inj₁)
    (large₂ : range (n ∘ m) ∈ f :=
          /-
            α : Type u
            β : Type v
            γ : Type u_1
            f✝ g : Ultrafilter α
            s t : Set α
            p q : α → Prop
            f : Ultrafilter γ
            m : α → β
            n : β → γ
            inj₀ : Function.Injective n
            large₀ : Membership.mem f (Set.range n)
            inj₁ : Function.Injective m
            large₁ : Membership.mem (f.comap inj₀ large₀) (Set.range m)
            inj₂ : optParam (Function.Injective (Function.comp n m)) ⋯
            ⊢ Membership.mem f (Set.range (Function.comp n m))
          -/
      (by rw [range_comp]; exact image_mem_of_mem_comap large₀ large₁)) :
                           /-
                             🎉 no goals
                           -/
    (f.comap inj₀ large₀).comap inj₁ large₁ = f.comap inj₂ large₂ :=
  coe_injective comap_comap


/-- The principal ultrafilter associated to a point `x`. -/
instance : Pure Ultrafilter :=
                                                  /-
                                                    α : Type u
                                                    β : Type v
                                                    γ : Type u_1
                                                    f g : Ultrafilter α
                                                    s✝ t : Set α
                                                    p q : α → Prop
                                                    α✝ : Type ?u.28228
                                                    a : α✝
                                                    s : Set α✝
                                                    ⊢ Iff (Not (Membership.mem (Pure.pure a) (HasCompl.compl s))) (Membership.mem  …
                                                  -/
  ⟨fun a => ofComplNotMemIff (pure a) fun s => by simp⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem mem_pure {a : α} {s : Set α} : s ∈ (pure a : Ultrafilter α) ↔ a ∈ s :=
  Iff.rfl


@[simp]
theorem coe_pure (a : α) : ↑(pure a : Ultrafilter α) = (pure a : Filter α) :=
  rfl


@[simp]
theorem map_pure (m : α → β) (a : α) : map m (pure a) = pure (m a) :=
  rfl


@[simp]
theorem comap_pure {m : α → β} (a : α) (inj : Injective m) (large) :
    comap (pure <| m a) inj large = pure a :=
  coe_injective <|
    Filter.comap_pure.trans <| by
      /-
        α : Type u
        β : Type v
        m : α → β
        a : α
        inj : Function.Injective m
        large : Membership.mem (Pure.pure (m a)) (Set.range m)
        ⊢ Eq (Filter.principal (Set.preimage m (Singleton.singleton (m a)))) ↑(Pure.pu …
      -/
      rw [coe_pure, ← principal_singleton, ← image_singleton, preimage_image_eq _ inj]
      /-
        🎉 no goals
      -/


theorem pure_injective : Injective (pure : α → Ultrafilter α) := fun _ _ h =>
  Filter.pure_injective (congr_arg Ultrafilter.toFilter h : _)


instance [Inhabited α] : Inhabited (Ultrafilter α) :=
  ⟨pure default⟩


instance [Nonempty α] : Nonempty (Ultrafilter α) :=
  Nonempty.map pure inferInstance


theorem eq_pure_of_finite_mem (h : s.Finite) (h' : s ∈ f) : ∃ x ∈ s, f = pure x := by
  /-
    α : Type u
    f : Ultrafilter α
    s : Set α
    h : s.Finite
    h' : Membership.mem f s
    ⊢ Exists fun x => And (Membership.mem s x) (Eq f (Pure.pure x))
  -/
  rw [← biUnion_of_singleton s] at h'
  /-
    α : Type u
    f : Ultrafilter α
    s : Set α
    h : s.Finite
    h' : Membership.mem f (Set.iUnion fun x => Set.iUnion fun h => Singleton.singl …
    ⊢ Exists fun x => And (Membership.mem s x) (Eq f (Pure.pure x))
  -/
  rcases (Ultrafilter.finite_biUnion_mem_iff h).mp h' with ⟨a, has, haf⟩
  /-
    case intro.intro
    α : Type u
    f : Ultrafilter α
    s : Set α
    h : s.Finite
    h' : Membership.mem f (Set.iUnion fun x => Set.iUnion fun h => Singleton.singl …
    a : α
    has : Membership.mem s a
    haf : Membership.mem f (Singleton.singleton a)
    ⊢ Exists fun x => And (Membership.mem s x) (Eq f (Pure.pure x))
  -/
  exact ⟨a, has, eq_of_le (Filter.le_pure_iff.2 haf)⟩
  /-
    🎉 no goals
  -/


theorem eq_pure_of_finite [Finite α] (f : Ultrafilter α) : ∃ a, f = pure a :=
  (eq_pure_of_finite_mem finite_univ univ_mem).imp fun _ ⟨_, ha⟩ => ha


theorem le_cofinite_or_eq_pure (f : Ultrafilter α) : (f : Filter α) ≤ cofinite ∨ ∃ a, f = pure a :=
  or_iff_not_imp_left.2 fun h =>
    let ⟨_, hs, hfin⟩ := Filter.disjoint_cofinite_right.1 (disjoint_iff_not_le.2 h)
    let ⟨a, _, hf⟩ := eq_pure_of_finite_mem hfin hs
    ⟨a, hf⟩


/-- Monadic bind for ultrafilters, coming from the one on filters
defined in terms of map and join. -/
def bind (f : Ultrafilter α) (m : α → Ultrafilter β) : Ultrafilter β :=
  ofComplNotMemIff (Filter.bind ↑f fun x => ↑(m x)) fun s => by
    /-
      α : Type u
      β : Type v
      γ : Type u_1
      f✝ g : Ultrafilter α
      s✝ t : Set α
      p q : α → Prop
      f : Ultrafilter α
      m : α → Ultrafilter β
      s : Set β
      ⊢ Iff (Not (Membership.mem ((↑f).bind fun x => ↑(m x)) (HasCompl.compl s))) (M …
    -/
    simp only [mem_bind', mem_coe, ← compl_mem_iff_not_mem, compl_setOf, compl_compl]
    /-
      🎉 no goals
    -/


instance instBind : Bind Ultrafilter :=
  ⟨@Ultrafilter.bind⟩


instance functor : Functor Ultrafilter where map := @Ultrafilter.map


instance monad : Monad Ultrafilter where map := @Ultrafilter.map


instance lawfulMonad : LawfulMonad Ultrafilter where
  id_map f := coe_injective (id_map f.toFilter)
  pure_bind a f := coe_injective (Filter.pure_bind a ((Ultrafilter.toFilter) ∘ f))
  bind_assoc _ _ _ := coe_injective (filter_eq rfl)
  bind_pure_comp f x := coe_injective (bind_pure_comp f x.1)
  map_const := rfl
  seqLeft_eq _ _ := rfl
  seqRight_eq _ _ := rfl
  pure_seq _ _ := rfl
  bind_map _ _ := rfl


/-- The ultrafilter lemma: Any proper filter is contained in an ultrafilter. -/
theorem exists_le (f : Filter α) [h : NeBot f] : ∃ u : Ultrafilter α, ↑u ≤ f :=
  let ⟨u, hu, huf⟩ := (eq_bot_or_exists_atom_le f).resolve_left h.ne
  ⟨ofAtom u hu, huf⟩


alias _root_.Filter.exists_ultrafilter_le := exists_le


/-- Construct an ultrafilter extending a given filter.
  The ultrafilter lemma is the assertion that such a filter exists;
  we use the axiom of choice to pick one. -/
noncomputable def of (f : Filter α) [NeBot f] : Ultrafilter α :=
  Classical.choose (exists_le f)


theorem of_le (f : Filter α) [NeBot f] : ↑(of f) ≤ f :=
  Classical.choose_spec (exists_le f)


theorem of_coe (f : Ultrafilter α) : of ↑f = f :=
               /-
                 α : Type u
                 f : Ultrafilter α
                 ⊢ (↑(Ultrafilter.of ↑f)).NeBot
               -/
  coe_inj.1 <| f.unique (of_le f.toFilter)
               /-
                 🎉 no goals
               -/


theorem exists_ultrafilter_of_finite_inter_nonempty (S : Set (Set α))
    (cond : ∀ T : Finset (Set α), (↑T : Set (Set α)) ⊆ S → (⋂₀ (↑T : Set (Set α))).Nonempty) :
    ∃ F : Ultrafilter α, S ⊆ F.sets :=
  haveI : NeBot (generate S) :=
    generate_neBot_iff.2 fun _ hts ht =>
      ht.coe_toFinset ▸ cond ht.toFinset (ht.coe_toFinset.symm ▸ hts)
  ⟨of (generate S), fun _ ht => (of_le <| generate S) <| GenerateSets.basic ht⟩


theorem isAtom_pure : IsAtom (pure a : Filter α) :=
  (pure a : Ultrafilter α).isAtom


protected theorem NeBot.le_pure_iff (hf : f.NeBot) : f ≤ pure a ↔ f = pure a :=
   /-
     α : Type u
     f : Filter α
     a : α
     hf : f.NeBot
     h : LE.le f ↑(Pure.pure a)
     ⊢ f.NeBot
   -/
  ⟨Ultrafilter.unique (pure a), le_of_eq⟩
   /-
     🎉 no goals
   -/


protected theorem NeBot.eq_pure_iff (hf : f.NeBot) {x : α} :
    f = pure x ↔ {x} ∈ f := by
  /-
    α : Type u
    f : Filter α
    hf : f.NeBot
    x : α
    ⊢ Iff (Eq f (Pure.pure x)) (Membership.mem f (Singleton.singleton x))
  -/
  rw [← hf.le_pure_iff, le_pure_iff]
  /-
    🎉 no goals
  -/


lemma atTop_eq_pure_of_isTop [LinearOrder α] {x : α} (hx : IsTop x) :
    (atTop : Filter α) = pure x := by
  /-
    α : Type u
    inst✝ : LinearOrder α
    x : α
    hx : IsTop x
    ⊢ Eq Filter.atTop (Pure.pure x)
  -/
  have : Nonempty α := ⟨x⟩
  /-
    α : Type u
    inst✝ : LinearOrder α
    x : α
    hx : IsTop x
    this : Nonempty α
    ⊢ Eq Filter.atTop (Pure.pure x)
  -/
  apply atTop_neBot.eq_pure_iff.2
  /-
    α : Type u
    inst✝ : LinearOrder α
    x : α
    hx : IsTop x
    this : Nonempty α
    ⊢ Membership.mem Filter.atTop (Singleton.singleton x)
  -/
  convert Ici_mem_atTop x using 1
  /-
    case h.e'_5
    α : Type u
    inst✝ : LinearOrder α
    x : α
    hx : IsTop x
    this : Nonempty α
    ⊢ Eq (Singleton.singleton x) (Set.Ici x)
  -/
  exact (Ici_eq_singleton_iff_isTop.2 hx).symm
  /-
    🎉 no goals
  -/


lemma atBot_eq_pure_of_isBot [LinearOrder α] {x : α} (hx : IsBot x) :
    (atBot : Filter α) = pure x :=
  @atTop_eq_pure_of_isTop αᵒᵈ _ _ hx


@[simp]
theorem lt_pure_iff : f < pure a ↔ f = ⊥ :=
  isAtom_pure.lt_iff


theorem le_pure_iff' : f ≤ pure a ↔ f = ⊥ ∨ f = pure a :=
  isAtom_pure.le_iff


@[simp]
theorem Iic_pure (a : α) : Iic (pure a : Filter α) = {⊥, pure a} :=
  isAtom_pure.Iic_eq


theorem mem_iff_ultrafilter : s ∈ f ↔ ∀ g : Ultrafilter α, ↑g ≤ f → s ∈ g := by
  /-
    α : Type u
    f : Filter α
    s : Set α
    ⊢ Iff (Membership.mem f s) (∀ (g : Ultrafilter α), LE.le (↑g) f → Membership.m …
  -/
  refine ⟨fun hf g hg => hg hf, fun H => by_contra fun hf => ?_⟩
  /-
    α : Type u
    f : Filter α
    s : Set α
    H : ∀ (g : Ultrafilter α), LE.le (↑g) f → Membership.mem g s
    hf : Not (Membership.mem f s)
    ⊢ False
  -/
  set g : Filter (sᶜ : Set α) := comap (↑) f
  /-
    α : Type u
    f : Filter α
    s : Set α
    H : ∀ (g : Ultrafilter α), LE.le (↑g) f → Membership.mem g s
    hf : Not (Membership.mem f s)
    g : Filter ↑(HasCompl.compl s) := Filter.comap Subtype.val f
    ⊢ False
  -/
  haveI : NeBot g := comap_neBot_iff_compl_range.2 (by simpa [compl_setOf] )
  /-
    α : Type u
    f : Filter α
    s : Set α
    H : ∀ (g : Ultrafilter α), LE.le (↑g) f → Membership.mem g s
    hf : Not (Membership.mem f s)
    g : Filter ↑(HasCompl.compl s) := Filter.comap Subtype.val f
    this : g.NeBot
    ⊢ False
  -/
  simpa using H ((of g).map (↑)) (map_le_iff_le_comap.mpr (of_le g))
  /-
    🎉 no goals
  -/


theorem le_iff_ultrafilter {f₁ f₂ : Filter α} : f₁ ≤ f₂ ↔ ∀ g : Ultrafilter α, ↑g ≤ f₁ → ↑g ≤ f₂ :=
  ⟨fun h _ h₁ => h₁.trans h, fun h _ hs => mem_iff_ultrafilter.2 fun g hg => h g hg hs⟩


/-- A filter equals the intersection of all the ultrafilters which contain it. -/
theorem iSup_ultrafilter_le_eq (f : Filter α) :
    ⨆ (g : Ultrafilter α) (_ : g ≤ f), (g : Filter α) = f :=
                                   /-
                                     α : Type u
                                     f f' : Filter α
                                     ⊢ Iff (LE.le (iSup fun g => iSup fun x => ↑g) f') (LE.le f f')
                                   -/
  eq_of_forall_ge_iff fun f' => by simp only [iSup_le_iff, ← le_iff_ultrafilter]
                                   /-
                                     🎉 no goals
                                   -/


/-- The `tendsto` relation can be checked on ultrafilters. -/
theorem tendsto_iff_ultrafilter (f : α → β) (l₁ : Filter α) (l₂ : Filter β) :
    Tendsto f l₁ l₂ ↔ ∀ g : Ultrafilter α, ↑g ≤ l₁ → Tendsto f g l₂ := by
  /-
    α : Type u
    β : Type v
    f : α → β
    l₁ : Filter α
    l₂ : Filter β
    ⊢ Iff (Filter.Tendsto f l₁ l₂) (∀ (g : Ultrafilter α), LE.le (↑g) l₁ → Filter. …
  -/
  simpa only [tendsto_iff_comap] using le_iff_ultrafilter
  /-
    🎉 no goals
  -/


theorem exists_ultrafilter_iff {f : Filter α} : (∃ u : Ultrafilter α, ↑u ≤ f) ↔ NeBot f :=
  ⟨fun ⟨_, uf⟩ => neBot_of_le uf, fun h => @exists_ultrafilter_le _ _ h⟩


theorem forall_neBot_le_iff {g : Filter α} {p : Filter α → Prop} (hp : Monotone p) :
    (∀ f : Filter α, NeBot f → f ≤ g → p f) ↔ ∀ f : Ultrafilter α, ↑f ≤ g → p f := by
  /-
    α : Type u
    g : Filter α
    p : Filter α → Prop
    hp : Monotone p
    ⊢ Iff (∀ (f : Filter α), f.NeBot → LE.le f g → p f) (∀ (f : Ultrafilter α), LE …
  -/
  refine ⟨fun H f hf => H f f.neBot hf, ?_⟩
  /-
    α : Type u
    g : Filter α
    p : Filter α → Prop
    hp : Monotone p
    ⊢ (∀ (f : Ultrafilter α), LE.le (↑f) g → p ↑f) → ∀ (f : Filter α), f.NeBot → L …
  -/
  intro H f hf hfg
  /-
    α : Type u
    g : Filter α
    p : Filter α → Prop
    hp : Monotone p
    H : ∀ (f : Ultrafilter α), LE.le (↑f) g → p ↑f
    f : Filter α
    hf : f.NeBot
    hfg : LE.le f g
    ⊢ p f
  -/
  exact hp (of_le f) (H _ ((of_le f).trans hfg))
  /-
    🎉 no goals
  -/


/-- The ultrafilter extending the cofinite filter. -/
noncomputable def hyperfilter : Ultrafilter α :=
  Ultrafilter.of cofinite


theorem hyperfilter_le_cofinite : ↑(hyperfilter α) ≤ @cofinite α :=
  Ultrafilter.of_le cofinite


theorem _root_.Nat.hyperfilter_le_atTop : (hyperfilter ℕ).toFilter ≤ atTop :=
  hyperfilter_le_cofinite.trans_eq Nat.cofinite_eq_atTop


@[simp]
theorem bot_ne_hyperfilter : (⊥ : Filter α) ≠ hyperfilter α :=
  (NeBot.ne inferInstance).symm


theorem nmem_hyperfilter_of_finite {s : Set α} (hf : s.Finite) : s ∉ hyperfilter α := fun hy =>
  compl_not_mem hy <| hyperfilter_le_cofinite hf.compl_mem_cofinite


alias _root_.Set.Finite.nmem_hyperfilter := nmem_hyperfilter_of_finite


theorem compl_mem_hyperfilter_of_finite {s : Set α} (hf : Set.Finite s) : sᶜ ∈ hyperfilter α :=
  compl_mem_iff_not_mem.2 hf.nmem_hyperfilter


alias _root_.Set.Finite.compl_mem_hyperfilter := compl_mem_hyperfilter_of_finite


theorem mem_hyperfilter_of_finite_compl {s : Set α} (hf : Set.Finite sᶜ) : s ∈ hyperfilter α :=
  compl_compl s ▸ hf.compl_mem_hyperfilter


theorem comap_inf_principal_neBot_of_image_mem (h : m '' s ∈ g) : (Filter.comap m g ⊓ 𝓟 s).NeBot :=
  Filter.comap_inf_principal_neBot_of_image_mem g.neBot h


/-- Ultrafilter extending the inf of a comapped ultrafilter and a principal ultrafilter. -/
noncomputable def ofComapInfPrincipal (h : m '' s ∈ g) : Ultrafilter α :=
  @of _ (Filter.comap m g ⊓ 𝓟 s) (comap_inf_principal_neBot_of_image_mem h)


theorem ofComapInfPrincipal_mem (h : m '' s ∈ g) : s ∈ ofComapInfPrincipal h := by
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    ⊢ Membership.mem (Ultrafilter.ofComapInfPrincipal h) s
  -/
  let f := Filter.comap m g ⊓ 𝓟 s
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    f : Filter α := Min.min (Filter.comap m ↑g) (Filter.principal s)
    ⊢ Membership.mem (Ultrafilter.ofComapInfPrincipal h) s
  -/
  haveI : f.NeBot := comap_inf_principal_neBot_of_image_mem h
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    f : Filter α := Min.min (Filter.comap m ↑g) (Filter.principal s)
    this : f.NeBot
    ⊢ Membership.mem (Ultrafilter.ofComapInfPrincipal h) s
  -/
  have : s ∈ f := mem_inf_of_right (mem_principal_self s)
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    f : Filter α := Min.min (Filter.comap m ↑g) (Filter.principal s)
    this✝ : f.NeBot
    this : Membership.mem f s
    ⊢ Membership.mem (Ultrafilter.ofComapInfPrincipal h) s
  -/
  exact le_def.mp (of_le _) s this
  /-
    🎉 no goals
  -/


theorem ofComapInfPrincipal_eq_of_map (h : m '' s ∈ g) : (ofComapInfPrincipal h).map m = g := by
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    ⊢ Eq (Ultrafilter.map m (Ultrafilter.ofComapInfPrincipal h)) g
  -/
  let f := Filter.comap m g ⊓ 𝓟 s
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    f : Filter α := Min.min (Filter.comap m ↑g) (Filter.principal s)
    ⊢ Eq (Ultrafilter.map m (Ultrafilter.ofComapInfPrincipal h)) g
  -/
  haveI : f.NeBot := comap_inf_principal_neBot_of_image_mem h
  /-
    α : Type u
    β : Type v
    m : α → β
    s : Set α
    g : Ultrafilter β
    h : Membership.mem g (Set.image m s)
    f : Filter α := Min.min (Filter.comap m ↑g) (Filter.principal s)
    this : f.NeBot
    ⊢ Eq (Ultrafilter.map m (Ultrafilter.ofComapInfPrincipal h)) g
  -/
  apply eq_of_le
  calc
    Filter.map m (of f) ≤ Filter.map m f := map_mono (of_le _)
    _ ≤ (Filter.map m <| Filter.comap m g) ⊓ Filter.map m (𝓟 s) := map_inf_le
    _ = (Filter.map m <| Filter.comap m g) ⊓ (𝓟 <| m '' s) := by rw [map_principal]
    _ ≤ ↑g ⊓ (𝓟 <| m '' s) := inf_le_inf_right _ map_comap_le
    _ = ↑g := inf_of_le_left (le_principal_iff.mpr h)


theorem eq_of_le_pure {X : Type _} {α : Filter X} (hα : α.NeBot) {x y : X}
    (hx : α ≤ pure x) (hy : α ≤ pure y) : x = y :=
  Filter.pure_injective (hα.le_pure_iff.mp hx ▸ hα.le_pure_iff.mp hy)


