theorem prod_mem_prod (hs : s ∈ f) (ht : t ∈ g) : s ×ˢ t ∈ f ×ˢ g :=
  inter_mem_inf (preimage_mem_comap hs) (preimage_mem_comap ht)


theorem mem_prod_iff {s : Set (α × β)} {f : Filter α} {g : Filter β} :
    s ∈ f ×ˢ g ↔ ∃ t₁ ∈ f, ∃ t₂ ∈ g, t₁ ×ˢ t₂ ⊆ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set (Prod α β)
    f : Filter α
    g : Filter β
    ⊢ Iff (Membership.mem (SProd.sprod f g) s) (Exists fun t₁ => And (Membership.m …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s : Set (Prod α β)
      f : Filter α
      g : Filter β
      ⊢ Membership.mem (SProd.sprod f g) s → Exists fun t₁ => And (Membership.mem f  …
    -/
  · rintro ⟨t₁, ⟨s₁, hs₁, hts₁⟩, t₂, ⟨s₂, hs₂, hts₂⟩, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : Filter α
      g : Filter β
      t₁ : Set (Prod α β)
      s₁ : Set α
      hs₁ : Membership.mem f s₁
      hts₁ : HasSubset.Subset (Set.preimage Prod.fst s₁) t₁
      t₂ : Set (Prod α β)
      s₂ : Set β
      hs₂ : Membership.mem g s₂
      hts₂ : HasSubset.Subset (Set.preimage Prod.snd s₂) t₂
      ⊢ Exists fun t₁_1 => And (Membership.mem f t₁_1) (Exists fun t₂_1 => And (Memb …
    -/
    exact ⟨s₁, hs₁, s₂, hs₂, fun p ⟨h, h'⟩ => ⟨hts₁ h, hts₂ h'⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      s : Set (Prod α β)
      f : Filter α
      g : Filter β
      ⊢ (Exists fun t₁ => And (Membership.mem f t₁) (Exists fun t₂ => And (Membershi …
    -/
  · rintro ⟨t₁, ht₁, t₂, ht₂, h⟩
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      s : Set (Prod α β)
      f : Filter α
      g : Filter β
      t₁ : Set α
      ht₁ : Membership.mem f t₁
      t₂ : Set β
      ht₂ : Membership.mem g t₂
      h : HasSubset.Subset (SProd.sprod t₁ t₂) s
      ⊢ Membership.mem (SProd.sprod f g) s
    -/
    exact mem_inf_of_inter (preimage_mem_comap ht₁) (preimage_mem_comap ht₂) h
    /-
      🎉 no goals
    -/


@[simp]
theorem compl_diagonal_mem_prod {l₁ l₂ : Filter α} : (diagonal α)ᶜ ∈ l₁ ×ˢ l₂ ↔ Disjoint l₁ l₂ := by
  /-
    α : Type u_1
    l₁ l₂ : Filter α
    ⊢ Iff (Membership.mem (SProd.sprod l₁ l₂) (HasCompl.compl (Set.diagonal α))) ( …
  -/
  simp only [mem_prod_iff, Filter.disjoint_iff, prod_subset_compl_diagonal_iff_disjoint]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_mem_prod_iff [f.NeBot] [g.NeBot] : s ×ˢ t ∈ f ×ˢ g ↔ s ∈ f ∧ t ∈ g :=
  ⟨fun h =>
    let ⟨_s', hs', _t', ht', H⟩ := mem_prod_iff.1 h
    (prod_subset_prod_iff.1 H).elim
      (fun ⟨hs's, ht't⟩ => ⟨mem_of_superset hs' hs's, mem_of_superset ht' ht't⟩) fun h =>
      h.elim (fun hs'e => absurd hs'e (nonempty_of_mem hs').ne_empty) fun ht'e =>
        absurd ht'e (nonempty_of_mem ht').ne_empty,
    fun h => prod_mem_prod h.1 h.2⟩


theorem mem_prod_principal {s : Set (α × β)} :
    s ∈ f ×ˢ 𝓟 t ↔ { a | ∀ b ∈ t, (a, b) ∈ s } ∈ f := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    f : Filter α
    s : Set (Prod α β)
    ⊢ Iff (Membership.mem (SProd.sprod f (Filter.principal t)) s) (Membership.mem  …
  -/
  rw [← @exists_mem_subset_iff _ f, mem_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    f : Filter α
    s : Set (Prod α β)
    ⊢ Iff (Exists fun t₁ => And (Membership.mem f t₁) (Exists fun t₂ => And (Membe …
  -/
  refine exists_congr fun u => Iff.rfl.and ⟨?_, fun h => ⟨t, mem_principal_self t, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      t : Set β
      f : Filter α
      s : Set (Prod α β)
      u : Set α
      ⊢ (Exists fun t₂ => And (Membership.mem (Filter.principal t) t₂) (HasSubset.Su …
    -/
  · rintro ⟨v, v_in, hv⟩ a a_in b b_in
    /-
      case refine_1.intro.intro
      α : Type u_1
      β : Type u_2
      t : Set β
      f : Filter α
      s : Set (Prod α β)
      u : Set α
      v : Set β
      v_in : Membership.mem (Filter.principal t) v
      hv : HasSubset.Subset (SProd.sprod u v) s
      a : α
      a_in : Membership.mem u a
      b : β
      b_in : Membership.mem t b
      ⊢ Membership.mem s { fst := a, snd := b }
    -/
    exact hv (mk_mem_prod a_in <| v_in b_in)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      t : Set β
      f : Filter α
      s : Set (Prod α β)
      u : Set α
      h : HasSubset.Subset u (setOf fun a => ∀ (b : β), Membership.mem t b → Members …
      ⊢ HasSubset.Subset (SProd.sprod u t) s
    -/
  · rintro ⟨x, y⟩ ⟨hx, hy⟩
    /-
      case refine_2.mk.intro
      α : Type u_1
      β : Type u_2
      t : Set β
      f : Filter α
      s : Set (Prod α β)
      u : Set α
      h : HasSubset.Subset u (setOf fun a => ∀ (b : β), Membership.mem t b → Members …
      x : α
      y : β
      hx : Membership.mem u { fst := x, snd := y }.1
      hy : Membership.mem t { fst := x, snd := y }.2
      ⊢ Membership.mem s { fst := x, snd := y }
    -/
    exact h hx y hy
    /-
      🎉 no goals
    -/


theorem mem_prod_top {s : Set (α × β)} :
    s ∈ f ×ˢ (⊤ : Filter β) ↔ { a | ∀ b, (a, b) ∈ s } ∈ f := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    s : Set (Prod α β)
    ⊢ Iff (Membership.mem (SProd.sprod f Top.top) s) (Membership.mem f (setOf fun  …
  -/
  rw [← principal_univ, mem_prod_principal]
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    s : Set (Prod α β)
    ⊢ Iff (Membership.mem f (setOf fun a => ∀ (b : β), Membership.mem Set.univ b → …
  -/
  simp only [mem_univ, forall_true_left]
  /-
    🎉 no goals
  -/


theorem eventually_prod_principal_iff {p : α × β → Prop} {s : Set β} :
    (∀ᶠ x : α × β in f ×ˢ 𝓟 s, p x) ↔ ∀ᶠ x : α in f, ∀ y : β, y ∈ s → p (x, y) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    p : Prod α β → Prop
    s : Set β
    ⊢ Iff (Filter.Eventually (fun x => p x) (SProd.sprod f (Filter.principal s)))  …
  -/
  rw [eventually_iff, eventually_iff, mem_prod_principal]
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    p : Prod α β → Prop
    s : Set β
    ⊢ Iff (Membership.mem f (setOf fun a => ∀ (b : β), Membership.mem s b → Member …
  -/
  simp only [mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem comap_prod (f : α → β × γ) (b : Filter β) (c : Filter γ) :
    comap f (b ×ˢ c) = comap (Prod.fst ∘ f) b ⊓ comap (Prod.snd ∘ f) c := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    b : Filter β
    c : Filter γ
    ⊢ Eq (Filter.comap f (SProd.sprod b c)) (Min.min (Filter.comap (Function.comp  …
  -/
  erw [comap_inf, Filter.comap_comap, Filter.comap_comap]
  /-
    🎉 no goals
  -/


theorem comap_prodMap_prod (f : α → β) (g : γ → δ) (lb : Filter β) (ld : Filter δ) :
    comap (Prod.map f g) (lb ×ˢ ld) = comap f lb ×ˢ comap g ld := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : α → β
    g : γ → δ
    lb : Filter β
    ld : Filter δ
    ⊢ Eq (Filter.comap (Prod.map f g) (SProd.sprod lb ld)) (SProd.sprod (Filter.co …
  -/
  simp [prod_eq_inf, comap_comap, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem prod_top : f ×ˢ (⊤ : Filter β) = f.comap Prod.fst := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    ⊢ Eq (SProd.sprod f Top.top) (Filter.comap Prod.fst f)
  -/
  rw [prod_eq_inf, comap_top, inf_top_eq]
  /-
    🎉 no goals
  -/


theorem top_prod : (⊤ : Filter α) ×ˢ g = g.comap Prod.snd := by
  /-
    α : Type u_1
    β : Type u_2
    g : Filter β
    ⊢ Eq (SProd.sprod Top.top g) (Filter.comap Prod.snd g)
  -/
  rw [prod_eq_inf, comap_top, top_inf_eq]
  /-
    🎉 no goals
  -/


theorem sup_prod (f₁ f₂ : Filter α) (g : Filter β) : (f₁ ⊔ f₂) ×ˢ g = (f₁ ×ˢ g) ⊔ (f₂ ×ˢ g) := by
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : Filter α
    g : Filter β
    ⊢ Eq (SProd.sprod (Max.max f₁ f₂) g) (Max.max (SProd.sprod f₁ g) (SProd.sprod  …
  -/
  simp only [prod_eq_inf, comap_sup, inf_sup_right]
  /-
    🎉 no goals
  -/


theorem prod_sup (f : Filter α) (g₁ g₂ : Filter β) : f ×ˢ (g₁ ⊔ g₂) = (f ×ˢ g₁) ⊔ (f ×ˢ g₂) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g₁ g₂ : Filter β
    ⊢ Eq (SProd.sprod f (Max.max g₁ g₂)) (Max.max (SProd.sprod f g₁) (SProd.sprod  …
  -/
  simp only [prod_eq_inf, comap_sup, inf_sup_left]
  /-
    🎉 no goals
  -/


theorem eventually_prod_iff {p : α × β → Prop} :
    (∀ᶠ x in f ×ˢ g, p x) ↔
      ∃ pa : α → Prop, (∀ᶠ x in f, pa x) ∧ ∃ pb : β → Prop, (∀ᶠ y in g, pb y) ∧
        ∀ {x}, pa x → ∀ {y}, pb y → p (x, y) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    p : Prod α β → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (SProd.sprod f g)) (Exists fun pa => A …
  -/
  simpa only [Set.prod_subset_iff] using @mem_prod_iff α β p f g
  /-
    🎉 no goals
  -/


theorem tendsto_fst : Tendsto Prod.fst (f ×ˢ g) f :=
  tendsto_inf_left tendsto_comap


theorem tendsto_snd : Tendsto Prod.snd (f ×ˢ g) g :=
  tendsto_inf_right tendsto_comap


/-- If a function tends to a product `g ×ˢ h` of filters, then its first component tends to
`g`. See also `Filter.Tendsto.fst_nhds` for the special case of converging to a point in a
product of two topological spaces. -/
theorem Tendsto.fst {h : Filter γ} {m : α → β × γ} (H : Tendsto m f (g ×ˢ h)) :
    Tendsto (fun a ↦ (m a).1) f g :=
  tendsto_fst.comp H


/-- If a function tends to a product `g ×ˢ h` of filters, then its second component tends to
`h`. See also `Filter.Tendsto.snd_nhds` for the special case of converging to a point in a
product of two topological spaces. -/
theorem Tendsto.snd {h : Filter γ} {m : α → β × γ} (H : Tendsto m f (g ×ˢ h)) :
    Tendsto (fun a ↦ (m a).2) f h :=
  tendsto_snd.comp H


theorem Tendsto.prod_mk {h : Filter γ} {m₁ : α → β} {m₂ : α → γ}
    (h₁ : Tendsto m₁ f g) (h₂ : Tendsto m₂ f h) : Tendsto (fun x => (m₁ x, m₂ x)) f (g ×ˢ h) :=
  tendsto_inf.2 ⟨tendsto_comap_iff.2 h₁, tendsto_comap_iff.2 h₂⟩


theorem tendsto_prod_swap : Tendsto (Prod.swap : α × β → β × α) (f ×ˢ g) (g ×ˢ f) :=
  tendsto_snd.prod_mk tendsto_fst


theorem Eventually.prod_inl {la : Filter α} {p : α → Prop} (h : ∀ᶠ x in la, p x) (lb : Filter β) :
    ∀ᶠ x in la ×ˢ lb, p (x : α × β).1 :=
  tendsto_fst.eventually h


theorem Eventually.prod_inr {lb : Filter β} {p : β → Prop} (h : ∀ᶠ x in lb, p x) (la : Filter α) :
    ∀ᶠ x in la ×ˢ lb, p (x : α × β).2 :=
  tendsto_snd.eventually h


theorem Eventually.prod_mk {la : Filter α} {pa : α → Prop} (ha : ∀ᶠ x in la, pa x) {lb : Filter β}
    {pb : β → Prop} (hb : ∀ᶠ y in lb, pb y) : ∀ᶠ p in la ×ˢ lb, pa (p : α × β).1 ∧ pb p.2 :=
  (ha.prod_inl lb).and (hb.prod_inr la)


theorem EventuallyEq.prod_map {δ} {la : Filter α} {fa ga : α → γ} (ha : fa =ᶠ[la] ga)
    {lb : Filter β} {fb gb : β → δ} (hb : fb =ᶠ[lb] gb) :
    Prod.map fa fb =ᶠ[la ×ˢ lb] Prod.map ga gb :=
  (Eventually.prod_mk ha hb).mono fun _ h => Prod.ext h.1 h.2


theorem EventuallyLE.prod_map {δ} [LE γ] [LE δ] {la : Filter α} {fa ga : α → γ} (ha : fa ≤ᶠ[la] ga)
    {lb : Filter β} {fb gb : β → δ} (hb : fb ≤ᶠ[lb] gb) :
    Prod.map fa fb ≤ᶠ[la ×ˢ lb] Prod.map ga gb :=
  Eventually.prod_mk ha hb


theorem Eventually.curry {la : Filter α} {lb : Filter β} {p : α × β → Prop}
    (h : ∀ᶠ x in la ×ˢ lb, p x) : ∀ᶠ x in la, ∀ᶠ y in lb, p (x, y) := by
  /-
    α : Type u_1
    β : Type u_2
    la : Filter α
    lb : Filter β
    p : Prod α β → Prop
    h : Filter.Eventually (fun x => p x) (SProd.sprod la lb)
    ⊢ Filter.Eventually (fun x => Filter.Eventually (fun y => p { fst := x, snd := …
  -/
  rcases eventually_prod_iff.1 h with ⟨pa, ha, pb, hb, h⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    la : Filter α
    lb : Filter β
    p : Prod α β → Prop
    h✝ : Filter.Eventually (fun x => p x) (SProd.sprod la lb)
    pa : α → Prop
    ha : Filter.Eventually (fun x => pa x) la
    pb : β → Prop
    hb : Filter.Eventually (fun y => pb y) lb
    h : ∀ {x : α}, pa x → ∀ {y : β}, pb y → p { fst := x, snd := y }
    ⊢ Filter.Eventually (fun x => Filter.Eventually (fun y => p { fst := x, snd := …
  -/
  exact ha.mono fun a ha => hb.mono fun b hb => h ha hb
  /-
    🎉 no goals
  -/


protected lemma Frequently.uncurry {la : Filter α} {lb : Filter β} {p : α → β → Prop}
    (h : ∃ᶠ x in la, ∃ᶠ y in lb, p x y) : ∃ᶠ xy in la ×ˢ lb, p xy.1 xy.2 :=
                 /-
                   α : Type u_1
                   β : Type u_2
                   la : Filter α
                   lb : Filter β
                   p : α → β → Prop
                   h✝ : Filter.Frequently (fun x => Filter.Frequently (fun y => p x y) lb) la
                   h : Filter.Eventually (fun x => Not ((fun xy => p xy.1 xy.2) x)) (SProd.sprod  …
                   ⊢ Filter.Eventually (fun x => Not ((fun x => Filter.Frequently (fun y => p x y …
                 -/
  mt (fun h ↦ by simpa only [not_frequently] using h.curry) h
                 /-
                   🎉 no goals
                 -/


/-- A fact that is eventually true about all pairs `l ×ˢ l` is eventually true about
all diagonal pairs `(i, i)` -/
theorem Eventually.diag_of_prod {p : α × α → Prop} (h : ∀ᶠ i in f ×ˢ f, p i) :
    ∀ᶠ i in f, p (i, i) := by
  /-
    α : Type u_1
    f : Filter α
    p : Prod α α → Prop
    h : Filter.Eventually (fun i => p i) (SProd.sprod f f)
    ⊢ Filter.Eventually (fun i => p { fst := i, snd := i }) f
  -/
  obtain ⟨t, ht, s, hs, hst⟩ := eventually_prod_iff.1 h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    f : Filter α
    p : Prod α α → Prop
    h : Filter.Eventually (fun i => p i) (SProd.sprod f f)
    t : α → Prop
    ht : Filter.Eventually (fun x => t x) f
    s : α → Prop
    hs : Filter.Eventually (fun y => s y) f
    hst : ∀ {x : α}, t x → ∀ {y : α}, s y → p { fst := x, snd := y }
    ⊢ Filter.Eventually (fun i => p { fst := i, snd := i }) f
  -/
  apply (ht.and hs).mono fun x hx => hst hx.1 hx.2
  /-
    🎉 no goals
  -/


theorem Eventually.diag_of_prod_left {f : Filter α} {g : Filter γ} {p : (α × α) × γ → Prop} :
    (∀ᶠ x in (f ×ˢ f) ×ˢ g, p x) → ∀ᶠ x : α × γ in f ×ˢ g, p ((x.1, x.1), x.2) := by
  /-
    α : Type u_1
    γ : Type u_3
    f : Filter α
    g : Filter γ
    p : Prod (Prod α α) γ → Prop
    ⊢ Filter.Eventually (fun x => p x) (SProd.sprod (SProd.sprod f f) g) → Filter. …
  -/
  intro h
  /-
    α : Type u_1
    γ : Type u_3
    f : Filter α
    g : Filter γ
    p : Prod (Prod α α) γ → Prop
    h : Filter.Eventually (fun x => p x) (SProd.sprod (SProd.sprod f f) g)
    ⊢ Filter.Eventually (fun x => p { fst := { fst := x.1, snd := x.1 }, snd := x. …
  -/
  obtain ⟨t, ht, s, hs, hst⟩ := eventually_prod_iff.1 h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    γ : Type u_3
    f : Filter α
    g : Filter γ
    p : Prod (Prod α α) γ → Prop
    h : Filter.Eventually (fun x => p x) (SProd.sprod (SProd.sprod f f) g)
    t : Prod α α → Prop
    ht : Filter.Eventually (fun x => t x) (SProd.sprod f f)
    s : γ → Prop
    hs : Filter.Eventually (fun y => s y) g
    hst : ∀ {x : Prod α α}, t x → ∀ {y : γ}, s y → p { fst := x, snd := y }
    ⊢ Filter.Eventually (fun x => p { fst := { fst := x.1, snd := x.1 }, snd := x. …
  -/
  exact (ht.diag_of_prod.prod_mk hs).mono fun x hx => by simp only [hst hx.1 hx.2]
  /-
    🎉 no goals
  -/


theorem Eventually.diag_of_prod_right {f : Filter α} {g : Filter γ} {p : α × γ × γ → Prop} :
    (∀ᶠ x in f ×ˢ (g ×ˢ g), p x) → ∀ᶠ x : α × γ in f ×ˢ g, p (x.1, x.2, x.2) := by
  /-
    α : Type u_1
    γ : Type u_3
    f : Filter α
    g : Filter γ
    p : Prod α (Prod γ γ) → Prop
    ⊢ Filter.Eventually (fun x => p x) (SProd.sprod f (SProd.sprod g g)) → Filter. …
  -/
  intro h
  /-
    α : Type u_1
    γ : Type u_3
    f : Filter α
    g : Filter γ
    p : Prod α (Prod γ γ) → Prop
    h : Filter.Eventually (fun x => p x) (SProd.sprod f (SProd.sprod g g))
    ⊢ Filter.Eventually (fun x => p { fst := x.1, snd := { fst := x.2, snd := x.2  …
  -/
  obtain ⟨t, ht, s, hs, hst⟩ := eventually_prod_iff.1 h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    γ : Type u_3
    f : Filter α
    g : Filter γ
    p : Prod α (Prod γ γ) → Prop
    h : Filter.Eventually (fun x => p x) (SProd.sprod f (SProd.sprod g g))
    t : α → Prop
    ht : Filter.Eventually (fun x => t x) f
    s : Prod γ γ → Prop
    hs : Filter.Eventually (fun y => s y) (SProd.sprod g g)
    hst : ∀ {x : α}, t x → ∀ {y : Prod γ γ}, s y → p { fst := x, snd := y }
    ⊢ Filter.Eventually (fun x => p { fst := x.1, snd := { fst := x.2, snd := x.2  …
  -/
  exact (ht.prod_mk hs.diag_of_prod).mono fun x hx => by simp only [hst hx.1 hx.2]
  /-
    🎉 no goals
  -/


theorem tendsto_diag : Tendsto (fun i => (i, i)) f (f ×ˢ f) :=
  tendsto_iff_eventually.mpr fun _ hpr => hpr.diag_of_prod


theorem prod_iInf_left [Nonempty ι] {f : ι → Filter α} {g : Filter β} :
    (⨅ i, f i) ×ˢ g = ⨅ i, f i ×ˢ g := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_5
    inst✝ : Nonempty ι
    f : ι → Filter α
    g : Filter β
    ⊢ Eq (SProd.sprod (iInf fun i => f i) g) (iInf fun i => SProd.sprod (f i) g)
  -/
  simp only [prod_eq_inf, comap_iInf, iInf_inf]
  /-
    🎉 no goals
  -/


theorem prod_iInf_right [Nonempty ι] {f : Filter α} {g : ι → Filter β} :
    (f ×ˢ ⨅ i, g i) = ⨅ i, f ×ˢ g i := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort u_5
    inst✝ : Nonempty ι
    f : Filter α
    g : ι → Filter β
    ⊢ Eq (SProd.sprod f (iInf fun i => g i)) (iInf fun i => SProd.sprod f (g i))
  -/
  simp only [prod_eq_inf, comap_iInf, inf_iInf]
  /-
    🎉 no goals
  -/


@[mono, gcongr]
theorem prod_mono {f₁ f₂ : Filter α} {g₁ g₂ : Filter β} (hf : f₁ ≤ f₂) (hg : g₁ ≤ g₂) :
    f₁ ×ˢ g₁ ≤ f₂ ×ˢ g₂ :=
  inf_le_inf (comap_mono hf) (comap_mono hg)


@[gcongr]
theorem prod_mono_left (g : Filter β) {f₁ f₂ : Filter α} (hf : f₁ ≤ f₂) : f₁ ×ˢ g ≤ f₂ ×ˢ g :=
  Filter.prod_mono hf rfl.le


@[gcongr]
theorem prod_mono_right (f : Filter α) {g₁ g₂ : Filter β} (hf : g₁ ≤ g₂) : f ×ˢ g₁ ≤ f ×ˢ g₂ :=
  Filter.prod_mono rfl.le hf


theorem prod_comap_comap_eq.{u, v, w, x} {α₁ : Type u} {α₂ : Type v} {β₁ : Type w} {β₂ : Type x}
    {f₁ : Filter α₁} {f₂ : Filter α₂} {m₁ : β₁ → α₁} {m₂ : β₂ → α₂} :
    comap m₁ f₁ ×ˢ comap m₂ f₂ = comap (fun p : β₁ × β₂ => (m₁ p.1, m₂ p.2)) (f₁ ×ˢ f₂) := by
  /-
    α₁ : Type u
    α₂ : Type v
    β₁ : Type w
    β₂ : Type x
    f₁ : Filter α₁
    f₂ : Filter α₂
    m₁ : β₁ → α₁
    m₂ : β₂ → α₂
    ⊢ Eq (SProd.sprod (Filter.comap m₁ f₁) (Filter.comap m₂ f₂)) (Filter.comap (fu …
  -/
  simp only [prod_eq_inf, comap_comap, comap_inf, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem prod_comm' : f ×ˢ g = comap Prod.swap (g ×ˢ f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Eq (SProd.sprod f g) (Filter.comap Prod.swap (SProd.sprod g f))
  -/
  simp only [prod_eq_inf, comap_comap, Function.comp_def, inf_comm, Prod.swap, comap_inf]
  /-
    🎉 no goals
  -/


theorem prod_comm : f ×ˢ g = map (fun p : β × α => (p.2, p.1)) (g ×ˢ f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Eq (SProd.sprod f g) (Filter.map (fun p => { fst := p.2, snd := p.1 }) (SPro …
  -/
  rw [prod_comm', ← map_swap_eq_comap_swap]
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Eq (Functor.map Prod.swap (SProd.sprod g f)) (Filter.map (fun p => { fst :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_prod_iff_left {s : Set (α × β)} :
    s ∈ f ×ˢ g ↔ ∃ t ∈ f, ∀ᶠ y in g, ∀ x ∈ t, (x, y) ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    s : Set (Prod α β)
    ⊢ Iff (Membership.mem (SProd.sprod f g) s) (Exists fun t => And (Membership.me …
  -/
  simp only [mem_prod_iff, prod_subset_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    s : Set (Prod α β)
    ⊢ Iff (Exists fun t₁ => And (Membership.mem f t₁) (Exists fun t₂ => And (Membe …
  -/
  refine exists_congr fun _ => Iff.rfl.and <| Iff.trans ?_ exists_mem_subset_iff
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    s : Set (Prod α β)
    x✝ : Set α
    ⊢ Iff (Exists fun t₂ => And (Membership.mem g t₂) (∀ (x : α), Membership.mem x …
  -/
  exact exists_congr fun _ => Iff.rfl.and forall₂_swap
  /-
    🎉 no goals
  -/


theorem mem_prod_iff_right {s : Set (α × β)} :
    s ∈ f ×ˢ g ↔ ∃ t ∈ g, ∀ᶠ x in f, ∀ y ∈ t, (x, y) ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    s : Set (Prod α β)
    ⊢ Iff (Membership.mem (SProd.sprod f g) s) (Exists fun t => And (Membership.me …
  -/
  rw [prod_comm, mem_map, mem_prod_iff_left]; rfl
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem map_fst_prod (f : Filter α) (g : Filter β) [NeBot g] : map Prod.fst (f ×ˢ g) = f := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    inst✝ : g.NeBot
    ⊢ Eq (Filter.map Prod.fst (SProd.sprod f g)) f
  -/
  ext s
  simp only [mem_map, mem_prod_iff_left, mem_preimage, eventually_const, ← subset_def,
    exists_mem_subset_iff]


@[simp]
theorem map_snd_prod (f : Filter α) (g : Filter β) [NeBot f] : map Prod.snd (f ×ˢ g) = g := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    inst✝ : f.NeBot
    ⊢ Eq (Filter.map Prod.snd (SProd.sprod f g)) g
  -/
  rw [prod_comm, map_map]; apply map_fst_prod
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem prod_le_prod {f₁ f₂ : Filter α} {g₁ g₂ : Filter β} [NeBot f₁] [NeBot g₁] :
    f₁ ×ˢ g₁ ≤ f₂ ×ˢ g₂ ↔ f₁ ≤ f₂ ∧ g₁ ≤ g₂ :=
  ⟨fun h =>
    ⟨map_fst_prod f₁ g₁ ▸ tendsto_fst.mono_left h, map_snd_prod f₁ g₁ ▸ tendsto_snd.mono_left h⟩,
    fun h => prod_mono h.1 h.2⟩


@[simp]
theorem prod_inj {f₁ f₂ : Filter α} {g₁ g₂ : Filter β} [NeBot f₁] [NeBot g₁] :
    f₁ ×ˢ g₁ = f₂ ×ˢ g₂ ↔ f₁ = f₂ ∧ g₁ = g₂ := by
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : Filter α
    g₁ g₂ : Filter β
    inst✝¹ : f₁.NeBot
    inst✝ : g₁.NeBot
    ⊢ Iff (Eq (SProd.sprod f₁ g₁) (SProd.sprod f₂ g₂)) (And (Eq f₁ f₂) (Eq g₁ g₂))
  -/
  refine ⟨fun h => ?_, fun h => h.1 ▸ h.2 ▸ rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : Filter α
    g₁ g₂ : Filter β
    inst✝¹ : f₁.NeBot
    inst✝ : g₁.NeBot
    h : Eq (SProd.sprod f₁ g₁) (SProd.sprod f₂ g₂)
    ⊢ And (Eq f₁ f₂) (Eq g₁ g₂)
  -/
  have hle : f₁ ≤ f₂ ∧ g₁ ≤ g₂ := prod_le_prod.1 h.le
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : Filter α
    g₁ g₂ : Filter β
    inst✝¹ : f₁.NeBot
    inst✝ : g₁.NeBot
    h : Eq (SProd.sprod f₁ g₁) (SProd.sprod f₂ g₂)
    hle : And (LE.le f₁ f₂) (LE.le g₁ g₂)
    ⊢ And (Eq f₁ f₂) (Eq g₁ g₂)
  -/
  haveI := neBot_of_le hle.1; haveI := neBot_of_le hle.2
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : Filter α
    g₁ g₂ : Filter β
    inst✝¹ : f₁.NeBot
    inst✝ : g₁.NeBot
    h : Eq (SProd.sprod f₁ g₁) (SProd.sprod f₂ g₂)
    hle : And (LE.le f₁ f₂) (LE.le g₁ g₂)
    this✝ : f₂.NeBot
    this : g₂.NeBot
    ⊢ And (Eq f₁ f₂) (Eq g₁ g₂)
  -/
  exact ⟨hle.1.antisymm <| (prod_le_prod.1 h.ge).1, hle.2.antisymm <| (prod_le_prod.1 h.ge).2⟩
  /-
    🎉 no goals
  -/


theorem eventually_swap_iff {p : α × β → Prop} :
    (∀ᶠ x : α × β in f ×ˢ g, p x) ↔ ∀ᶠ y : β × α in g ×ˢ f, p y.swap := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    p : Prod α β → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (SProd.sprod f g)) (Filter.Eventually  …
  -/
  rw [prod_comm]; rfl
                  /-
                    🎉 no goals
                  -/


theorem prod_assoc (f : Filter α) (g : Filter β) (h : Filter γ) :
    map (Equiv.prodAssoc α β γ) ((f ×ˢ g) ×ˢ h) = f ×ˢ (g ×ˢ h) := by
  simp_rw [← comap_equiv_symm, prod_eq_inf, comap_inf, comap_comap, inf_assoc,
    Function.comp_def, Equiv.prodAssoc_symm_apply]


theorem prod_assoc_symm (f : Filter α) (g : Filter β) (h : Filter γ) :
    map (Equiv.prodAssoc α β γ).symm (f ×ˢ (g ×ˢ h)) = (f ×ˢ g) ×ˢ h := by
  simp_rw [map_equiv_symm, prod_eq_inf, comap_inf, comap_comap, inf_assoc,
    Function.comp_def, Equiv.prodAssoc_apply]


theorem tendsto_prodAssoc {h : Filter γ} :
    Tendsto (Equiv.prodAssoc α β γ) ((f ×ˢ g) ×ˢ h) (f ×ˢ (g ×ˢ h)) :=
  (prod_assoc f g h).le


theorem tendsto_prodAssoc_symm {h : Filter γ} :
    Tendsto (Equiv.prodAssoc α β γ).symm (f ×ˢ (g ×ˢ h)) ((f ×ˢ g) ×ˢ h) :=
  (prod_assoc_symm f g h).le


/-- A useful lemma when dealing with uniformities. -/
theorem map_swap4_prod {h : Filter γ} {k : Filter δ} :
    map (fun p : (α × β) × γ × δ => ((p.1.1, p.2.1), (p.1.2, p.2.2))) ((f ×ˢ g) ×ˢ (h ×ˢ k)) =
      (f ×ˢ h) ×ˢ (g ×ˢ k) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : Filter α
    g : Filter β
    h : Filter γ
    k : Filter δ
    ⊢ Eq (Filter.map (fun p => { fst := { fst := p.1.1, snd := p.2.1 }, snd := { f …
  -/
  simp_rw [map_swap4_eq_comap, prod_eq_inf, comap_inf, comap_comap]; ac_rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem tendsto_swap4_prod {h : Filter γ} {k : Filter δ} :
    Tendsto (fun p : (α × β) × γ × δ => ((p.1.1, p.2.1), (p.1.2, p.2.2))) ((f ×ˢ g) ×ˢ (h ×ˢ k))
      ((f ×ˢ h) ×ˢ (g ×ˢ k)) :=
  map_swap4_prod.le


theorem prod_map_map_eq.{u, v, w, x} {α₁ : Type u} {α₂ : Type v} {β₁ : Type w} {β₂ : Type x}
    {f₁ : Filter α₁} {f₂ : Filter α₂} {m₁ : α₁ → β₁} {m₂ : α₂ → β₂} :
    map m₁ f₁ ×ˢ map m₂ f₂ = map (fun p : α₁ × α₂ => (m₁ p.1, m₂ p.2)) (f₁ ×ˢ f₂) :=
  le_antisymm
    (fun s hs =>
      let ⟨s₁, hs₁, s₂, hs₂, h⟩ := mem_prod_iff.mp hs
      mem_of_superset (prod_mem_prod (image_mem_map hs₁) (image_mem_map hs₂)) <|
           /-
             α₁ : Type u
             α₂ : Type v
             β₁ : Type w
             β₂ : Type x
             f₁ : Filter α₁
             f₂ : Filter α₂
             m₁ : α₁ → β₁
             m₂ : α₂ → β₂
             s : Set (Prod β₁ β₂)
             hs : Membership.mem (SProd.sprod f₁ f₂) (Set.preimage (fun p => { fst := m₁ p. …
             s₁ : Set α₁
             hs₁ : Membership.mem f₁ s₁
             s₂ : Set α₂
             hs₂ : Membership.mem f₂ s₂
             h : HasSubset.Subset (SProd.sprod s₁ s₂) (Set.preimage (fun p => { fst := m₁ p …
             ⊢ HasSubset.Subset (SProd.sprod (Set.image m₁ s₁) (Set.image m₂ s₂)) s
           -/
        by rwa [prod_image_image_eq, image_subset_iff])
           /-
             🎉 no goals
           -/
    ((tendsto_map.comp tendsto_fst).prod_mk (tendsto_map.comp tendsto_snd))


theorem prod_map_map_eq' {α₁ : Type*} {α₂ : Type*} {β₁ : Type*} {β₂ : Type*} (f : α₁ → α₂)
    (g : β₁ → β₂) (F : Filter α₁) (G : Filter β₁) :
    map f F ×ˢ map g G = map (Prod.map f g) (F ×ˢ G) :=
  prod_map_map_eq


theorem prod_map_left (f : α → β) (F : Filter α) (G : Filter γ) :
    map f F ×ˢ G = map (Prod.map f id) (F ×ˢ G) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    F : Filter α
    G : Filter γ
    ⊢ Eq (SProd.sprod (Filter.map f F) G) (Filter.map (Prod.map f id) (SProd.sprod …
  -/
  rw [← prod_map_map_eq', map_id]
  /-
    🎉 no goals
  -/


theorem prod_map_right (f : β → γ) (F : Filter α) (G : Filter β) :
    F ×ˢ map f G = map (Prod.map id f) (F ×ˢ G) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : β → γ
    F : Filter α
    G : Filter β
    ⊢ Eq (SProd.sprod F (Filter.map f G)) (Filter.map (Prod.map id f) (SProd.sprod …
  -/
  rw [← prod_map_map_eq', map_id]
  /-
    🎉 no goals
  -/


theorem le_prod_map_fst_snd {f : Filter (α × β)} : f ≤ map Prod.fst f ×ˢ map Prod.snd f :=
  le_inf le_comap_map le_comap_map


theorem Tendsto.prod_map {δ : Type*} {f : α → γ} {g : β → δ} {a : Filter α} {b : Filter β}
    {c : Filter γ} {d : Filter δ} (hf : Tendsto f a c) (hg : Tendsto g b d) :
    Tendsto (Prod.map f g) (a ×ˢ b) (c ×ˢ d) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_6
    f : α → γ
    g : β → δ
    a : Filter α
    b : Filter β
    c : Filter γ
    d : Filter δ
    hf : Filter.Tendsto f a c
    hg : Filter.Tendsto g b d
    ⊢ Filter.Tendsto (Prod.map f g) (SProd.sprod a b) (SProd.sprod c d)
  -/
  erw [Tendsto, ← prod_map_map_eq]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_6
    f : α → γ
    g : β → δ
    a : Filter α
    b : Filter β
    c : Filter γ
    d : Filter δ
    hf : Filter.Tendsto f a c
    hg : Filter.Tendsto g b d
    ⊢ LE.le (SProd.sprod (Filter.map f a) (Filter.map g b)) (SProd.sprod c d)
  -/
  exact Filter.prod_mono hf hg
  /-
    🎉 no goals
  -/


protected theorem map_prod (m : α × β → γ) (f : Filter α) (g : Filter β) :
    map m (f ×ˢ g) = (f.map fun a b => m (a, b)).seq g := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : Prod α β → γ
    f : Filter α
    g : Filter β
    ⊢ Eq (Filter.map m (SProd.sprod f g)) ((Filter.map (fun a b => m { fst := a, s …
  -/
  simp only [Filter.ext_iff, mem_map, mem_prod_iff, mem_map_seq_iff, exists_and_left]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : Prod α β → γ
    f : Filter α
    g : Filter β
    ⊢ ∀ (s : Set γ), Iff (Exists fun t₁ => And (Membership.mem f t₁) (Exists fun t …
  -/
  intro s
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : Prod α β → γ
    f : Filter α
    g : Filter β
    s : Set γ
    ⊢ Iff (Exists fun t₁ => And (Membership.mem f t₁) (Exists fun t₂ => And (Membe …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : Prod α β → γ
      f : Filter α
      g : Filter β
      s : Set γ
      ⊢ (Exists fun t₁ => And (Membership.mem f t₁) (Exists fun t₂ => And (Membershi …
    -/
  · exact fun ⟨t, ht, s, hs, h⟩ => ⟨s, hs, t, ht, fun x hx y hy => @h ⟨x, y⟩ ⟨hx, hy⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : Prod α β → γ
      f : Filter α
      g : Filter β
      s : Set γ
      ⊢ (Exists fun t => And (Membership.mem g t) (Exists fun x => And (Membership.m …
    -/
  · exact fun ⟨s, hs, t, ht, h⟩ => ⟨t, ht, s, hs, fun ⟨x, y⟩ ⟨hx, hy⟩ => h x hx y hy⟩
    /-
      🎉 no goals
    -/


theorem prod_eq : f ×ˢ g = (f.map Prod.mk).seq g := f.map_prod id g


theorem prod_inf_prod {f₁ f₂ : Filter α} {g₁ g₂ : Filter β} :
    (f₁ ×ˢ g₁) ⊓ (f₂ ×ˢ g₂) = (f₁ ⊓ f₂) ×ˢ (g₁ ⊓ g₂) := by
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : Filter α
    g₁ g₂ : Filter β
    ⊢ Eq (Min.min (SProd.sprod f₁ g₁) (SProd.sprod f₂ g₂)) (SProd.sprod (Min.min f …
  -/
  simp only [prod_eq_inf, comap_inf, inf_comm, inf_assoc, inf_left_comm]
  /-
    🎉 no goals
  -/


theorem inf_prod {f₁ f₂ : Filter α} : (f₁ ⊓ f₂) ×ˢ g = (f₁ ×ˢ g) ⊓ (f₂ ×ˢ g) := by
  /-
    α : Type u_1
    β : Type u_2
    g : Filter β
    f₁ f₂ : Filter α
    ⊢ Eq (SProd.sprod (Min.min f₁ f₂) g) (Min.min (SProd.sprod f₁ g) (SProd.sprod  …
  -/
  rw [prod_inf_prod, inf_idem]
  /-
    🎉 no goals
  -/


theorem prod_inf {g₁ g₂ : Filter β} : f ×ˢ (g₁ ⊓ g₂) = (f ×ˢ g₁) ⊓ (f ×ˢ g₂) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g₁ g₂ : Filter β
    ⊢ Eq (SProd.sprod f (Min.min g₁ g₂)) (Min.min (SProd.sprod f g₁) (SProd.sprod  …
  -/
  rw [prod_inf_prod, inf_idem]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_principal_principal {s : Set α} {t : Set β} : 𝓟 s ×ˢ 𝓟 t = 𝓟 (s ×ˢ t) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    ⊢ Eq (SProd.sprod (Filter.principal s) (Filter.principal t)) (Filter.principal …
  -/
  simp only [prod_eq_inf, comap_principal, principal_eq_iff_eq, comap_principal, inf_principal]; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


@[simp]
theorem pure_prod {a : α} {f : Filter β} : pure a ×ˢ f = map (Prod.mk a) f := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    f : Filter β
    ⊢ Eq (SProd.sprod (Pure.pure a) f) (Filter.map (Prod.mk a) f)
  -/
  rw [prod_eq, map_pure, pure_seq_eq_map]
  /-
    🎉 no goals
  -/


theorem map_pure_prod (f : α → β → γ) (a : α) (B : Filter β) :
    map (Function.uncurry f) (pure a ×ˢ B) = map (f a) B := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    a : α
    B : Filter β
    ⊢ Eq (Filter.map (Function.uncurry f) (SProd.sprod (Pure.pure a) B)) (Filter.m …
  -/
  rw [Filter.pure_prod]; rfl
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem prod_pure {b : β} : f ×ˢ pure b = map (fun a => (a, b)) f := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    b : β
    ⊢ Eq (SProd.sprod f (Pure.pure b)) (Filter.map (fun a => { fst := a, snd := b  …
  -/
  rw [prod_eq, seq_pure, map_map]; rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem prod_pure_pure {a : α} {b : β} :
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type u_2
                                                                     a : α
                                                                     b : β
                                                                     ⊢ Eq (SProd.sprod (Pure.pure a) (Pure.pure b)) (Pure.pure { fst := a, snd := b …
                                                                   -/
    (pure a : Filter α) ×ˢ (pure b : Filter β) = pure (a, b) := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem prod_eq_bot : f ×ˢ g = ⊥ ↔ f = ⊥ ∨ g = ⊥ := by
  simp_rw [← empty_mem_iff_bot, mem_prod_iff, subset_empty_iff, prod_eq_empty_iff, ← exists_prop,
    Subtype.exists', exists_or, exists_const, Subtype.exists, exists_prop, exists_eq_right]


@[simp] theorem prod_bot : f ×ˢ (⊥ : Filter β) = ⊥ := prod_eq_bot.2 <| Or.inr rfl


@[simp] theorem bot_prod : (⊥ : Filter α) ×ˢ g = ⊥ := prod_eq_bot.2 <| Or.inl rfl


theorem prod_neBot : NeBot (f ×ˢ g) ↔ NeBot f ∧ NeBot g := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Iff (SProd.sprod f g).NeBot (And f.NeBot g.NeBot)
  -/
  simp only [neBot_iff, Ne, prod_eq_bot, not_or]
  /-
    🎉 no goals
  -/


protected theorem NeBot.prod (hf : NeBot f) (hg : NeBot g) : NeBot (f ×ˢ g) := prod_neBot.2 ⟨hf, hg⟩


instance prod.instNeBot [hf : NeBot f] [hg : NeBot g] : NeBot (f ×ˢ g) := hf.prod hg


@[simp]
lemma disjoint_prod {f' : Filter α} {g' : Filter β} :
    Disjoint (f ×ˢ g) (f' ×ˢ g') ↔ Disjoint f f' ∨ Disjoint g g' := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    f' : Filter α
    g' : Filter β
    ⊢ Iff (Disjoint (SProd.sprod f g) (SProd.sprod f' g')) (Or (Disjoint f f') (Di …
  -/
  simp only [disjoint_iff, prod_inf_prod, prod_eq_bot]
  /-
    🎉 no goals
  -/


/-- `p ∧ q` occurs frequently along the product of two filters
iff both `p` and `q` occur frequently along the corresponding filters. -/
theorem frequently_prod_and {p : α → Prop} {q : β → Prop} :
    (∃ᶠ x in f ×ˢ g, p x.1 ∧ q x.2) ↔ (∃ᶠ a in f, p a) ∧ ∃ᶠ b in g, q b := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    p : α → Prop
    q : β → Prop
    ⊢ Iff (Filter.Frequently (fun x => And (p x.1) (q x.2)) (SProd.sprod f g)) (An …
  -/
  simp only [frequently_iff_neBot, ← prod_neBot, ← prod_inf_prod, prod_principal_principal]
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    p : α → Prop
    q : β → Prop
    ⊢ Iff (Min.min (SProd.sprod f g) (Filter.principal (setOf fun x => And (p x.1) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tendsto_prod_iff {f : α × β → γ} {x : Filter α} {y : Filter β} {z : Filter γ} :
    Tendsto f (x ×ˢ y) z ↔ ∀ W ∈ z, ∃ U ∈ x, ∃ V ∈ y, ∀ x y, x ∈ U → y ∈ V → f (x, y) ∈ W := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : Prod α β → γ
    x : Filter α
    y : Filter β
    z : Filter γ
    ⊢ Iff (Filter.Tendsto f (SProd.sprod x y) z) (∀ (W : Set γ), Membership.mem z  …
  -/
  simp only [tendsto_def, mem_prod_iff, prod_sub_preimage_iff, exists_prop]
  /-
    🎉 no goals
  -/


theorem tendsto_prod_iff' {g' : Filter γ} {s : α → β × γ} :
    Tendsto s f (g ×ˢ g') ↔ Tendsto (fun n => (s n).1) f g ∧ Tendsto (fun n => (s n).2) f g' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : Filter α
    g : Filter β
    g' : Filter γ
    s : α → Prod β γ
    ⊢ Iff (Filter.Tendsto s f (SProd.sprod g g')) (And (Filter.Tendsto (fun n => ( …
  -/
  simp only [prod_eq_inf, tendsto_inf, tendsto_comap_iff, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem le_prod {f : Filter (α × β)} {g : Filter α} {g' : Filter β} :
    (f ≤ g ×ˢ g') ↔ Tendsto Prod.fst f g ∧ Tendsto Prod.snd f g' :=
  tendsto_prod_iff'


theorem coprod_eq_prod_top_sup_top_prod (f : Filter α) (g : Filter β) :
    Filter.coprod f g = f ×ˢ ⊤ ⊔ ⊤ ×ˢ g := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Eq (f.coprod g) (Max.max (SProd.sprod f Top.top) (SProd.sprod Top.top g))
  -/
  rw [prod_top, top_prod]
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Eq (f.coprod g) (Max.max (Filter.comap Prod.fst f) (Filter.comap Prod.snd g))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_coprod_iff {s : Set (α × β)} {f : Filter α} {g : Filter β} :
    s ∈ f.coprod g ↔ (∃ t₁ ∈ f, Prod.fst ⁻¹' t₁ ⊆ s) ∧ ∃ t₂ ∈ g, Prod.snd ⁻¹' t₂ ⊆ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set (Prod α β)
    f : Filter α
    g : Filter β
    ⊢ Iff (Membership.mem (f.coprod g) s) (And (Exists fun t₁ => And (Membership.m …
  -/
  simp [Filter.coprod]
  /-
    🎉 no goals
  -/


@[simp]
theorem bot_coprod (l : Filter β) : (⊥ : Filter α).coprod l = comap Prod.snd l := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter β
    ⊢ Eq (Bot.bot.coprod l) (Filter.comap Prod.snd l)
  -/
  simp [Filter.coprod]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprod_bot (l : Filter α) : l.coprod (⊥ : Filter β) = comap Prod.fst l := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    ⊢ Eq (l.coprod Bot.bot) (Filter.comap Prod.fst l)
  -/
  simp [Filter.coprod]
  /-
    🎉 no goals
  -/


                                                                        /-
                                                                          α : Type u_1
                                                                          β : Type u_2
                                                                          ⊢ Eq (Bot.bot.coprod Bot.bot) Bot.bot
                                                                        -/
theorem bot_coprod_bot : (⊥ : Filter α).coprod (⊥ : Filter β) = ⊥ := by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem compl_mem_coprod {s : Set (α × β)} {la : Filter α} {lb : Filter β} :
    sᶜ ∈ la.coprod lb ↔ (Prod.fst '' s)ᶜ ∈ la ∧ (Prod.snd '' s)ᶜ ∈ lb := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set (Prod α β)
    la : Filter α
    lb : Filter β
    ⊢ Iff (Membership.mem (la.coprod lb) (HasCompl.compl s)) (And (Membership.mem  …
  -/
  simp only [Filter.coprod, mem_sup, compl_mem_comap]
  /-
    🎉 no goals
  -/


@[mono]
theorem coprod_mono {f₁ f₂ : Filter α} {g₁ g₂ : Filter β} (hf : f₁ ≤ f₂) (hg : g₁ ≤ g₂) :
    f₁.coprod g₁ ≤ f₂.coprod g₂ :=
  sup_le_sup (comap_mono hf) (comap_mono hg)


theorem coprod_neBot_iff : (f.coprod g).NeBot ↔ f.NeBot ∧ Nonempty β ∨ Nonempty α ∧ g.NeBot := by
  /-
    α : Type u_1
    β : Type u_2
    f : Filter α
    g : Filter β
    ⊢ Iff (f.coprod g).NeBot (Or (And f.NeBot (Nonempty β)) (And (Nonempty α) g.Ne …
  -/
  simp [Filter.coprod]
  /-
    🎉 no goals
  -/


@[instance]
theorem coprod_neBot_left [NeBot f] [Nonempty β] : (f.coprod g).NeBot :=
  coprod_neBot_iff.2 (Or.inl ⟨‹_›, ‹_›⟩)


@[instance]
theorem coprod_neBot_right [NeBot g] [Nonempty α] : (f.coprod g).NeBot :=
  coprod_neBot_iff.2 (Or.inr ⟨‹_›, ‹_›⟩)


theorem coprod_inf_prod_le (f₁ f₂ : Filter α) (g₁ g₂ : Filter β) :
    f₁.coprod g₁ ⊓ f₂ ×ˢ g₂ ≤ f₁ ×ˢ g₂ ⊔ f₂ ×ˢ g₁ := calc
  f₁.coprod g₁ ⊓ f₂ ×ˢ g₂
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        f₁ f₂ : Filter α
                                                        g₁ g₂ : Filter β
                                                        ⊢ Eq (Min.min (f₁.coprod g₁) (SProd.sprod f₂ g₂)) (Min.min (Max.max (SProd.spr …
                                                      -/
  _ = (f₁ ×ˢ ⊤ ⊔ ⊤ ×ˢ g₁) ⊓ f₂ ×ˢ g₂            := by rw [coprod_eq_prod_top_sup_top_prod]
                                                      /-
                                                        🎉 no goals
                                                      -/
  _ = f₁ ×ˢ ⊤ ⊓ f₂ ×ˢ g₂ ⊔ ⊤ ×ˢ g₁ ⊓ f₂ ×ˢ g₂   := inf_sup_right _ _ _
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        f₁ f₂ : Filter α
                                                        g₁ g₂ : Filter β
                                                        ⊢ Eq (Max.max (Min.min (SProd.sprod f₁ Top.top) (SProd.sprod f₂ g₂)) (Min.min  …
                                                      -/
  _ = (f₁ ⊓ f₂) ×ˢ g₂ ⊔ f₂ ×ˢ (g₁ ⊓ g₂)         := by simp [prod_inf_prod]
                                                      /-
                                                        🎉 no goals
                                                      -/
  _ ≤ f₁ ×ˢ g₂ ⊔ f₂ ×ˢ g₁                       :=
    sup_le_sup (prod_mono inf_le_left le_rfl) (prod_mono le_rfl inf_le_left)


theorem principal_coprod_principal (s : Set α) (t : Set β) :
    (𝓟 s).coprod (𝓟 t) = 𝓟 (sᶜ ×ˢ tᶜ)ᶜ := by
  rw [Filter.coprod, comap_principal, comap_principal, sup_principal, Set.prod_eq, compl_inter,
    preimage_compl, preimage_compl, compl_compl, compl_compl]

-- this inequality can be strict; see `map_const_principal_coprod_map_id_principal` and
-- `map_prod_map_const_id_principal_coprod_principal` below.

theorem map_prod_map_coprod_le.{u, v, w, x} {α₁ : Type u} {α₂ : Type v} {β₁ : Type w} {β₂ : Type x}
    {f₁ : Filter α₁} {f₂ : Filter α₂} {m₁ : α₁ → β₁} {m₂ : α₂ → β₂} :
    map (Prod.map m₁ m₂) (f₁.coprod f₂) ≤ (map m₁ f₁).coprod (map m₂ f₂) := by
  /-
    α₁ : Type u
    α₂ : Type v
    β₁ : Type w
    β₂ : Type x
    f₁ : Filter α₁
    f₂ : Filter α₂
    m₁ : α₁ → β₁
    m₂ : α₂ → β₂
    ⊢ LE.le (Filter.map (Prod.map m₁ m₂) (f₁.coprod f₂)) ((Filter.map m₁ f₁).copro …
  -/
  intro s
  /-
    α₁ : Type u
    α₂ : Type v
    β₁ : Type w
    β₂ : Type x
    f₁ : Filter α₁
    f₂ : Filter α₂
    m₁ : α₁ → β₁
    m₂ : α₂ → β₂
    s : Set (Prod β₁ β₂)
    ⊢ Membership.mem ((Filter.map m₁ f₁).coprod (Filter.map m₂ f₂)) s → Membership …
  -/
  simp only [mem_map, mem_coprod_iff]
  /-
    α₁ : Type u
    α₂ : Type v
    β₁ : Type w
    β₂ : Type x
    f₁ : Filter α₁
    f₂ : Filter α₂
    m₁ : α₁ → β₁
    m₂ : α₂ → β₂
    s : Set (Prod β₁ β₂)
    ⊢ And (Exists fun t₁ => And (Membership.mem f₁ (Set.preimage m₁ t₁)) (HasSubse …
  -/
  rintro ⟨⟨u₁, hu₁, h₁⟩, u₂, hu₂, h₂⟩
  /-
    case intro.intro.intro.intro.intro
    α₁ : Type u
    α₂ : Type v
    β₁ : Type w
    β₂ : Type x
    f₁ : Filter α₁
    f₂ : Filter α₂
    m₁ : α₁ → β₁
    m₂ : α₂ → β₂
    s : Set (Prod β₁ β₂)
    u₁ : Set β₁
    hu₁ : Membership.mem f₁ (Set.preimage m₁ u₁)
    h₁ : HasSubset.Subset (Set.preimage Prod.fst u₁) s
    u₂ : Set β₂
    hu₂ : Membership.mem f₂ (Set.preimage m₂ u₂)
    h₂ : HasSubset.Subset (Set.preimage Prod.snd u₂) s
    ⊢ And (Exists fun t₁ => And (Membership.mem f₁ t₁) (HasSubset.Subset (Set.prei …
  -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  refine ⟨⟨m₁ ⁻¹' u₁, hu₁, fun _ hx => h₁ ?_⟩, ⟨m₂ ⁻¹' u₂, hu₂, fun _ hx => h₂ ?_⟩⟩ <;> convert hx
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


/-- Characterization of the coproduct of the `Filter.map`s of two principal filters `𝓟 {a}` and
`𝓟 {i}`, the first under the constant function `fun a => b` and the second under the identity
function. Together with the next lemma, `map_prod_map_const_id_principal_coprod_principal`, this
provides an example showing that the inequality in the lemma `map_prod_map_coprod_le` can be strict.
-/
theorem map_const_principal_coprod_map_id_principal {α β ι : Type*} (a : α) (b : β) (i : ι) :
    (map (fun _ => b) (𝓟 {a})).coprod (map id (𝓟 {i})) =
      𝓟 ((({b} : Set β) ×ˢ univ) ∪ (univ ×ˢ ({i} : Set ι))) := by
  simp only [map_principal, Filter.coprod, comap_principal, sup_principal, image_singleton,
    image_id, prod_univ, univ_prod, id]


/-- Characterization of the `Filter.map` of the coproduct of two principal filters `𝓟 {a}` and
`𝓟 {i}`, under the `Prod.map` of two functions, respectively the constant function `fun a => b` and
the identity function.  Together with the previous lemma,
`map_const_principal_coprod_map_id_principal`, this provides an example showing that the inequality
in the lemma `map_prod_map_coprod_le` can be strict. -/
theorem map_prod_map_const_id_principal_coprod_principal {α β ι : Type*} (a : α) (b : β) (i : ι) :
    map (Prod.map (fun _ : α => b) id) ((𝓟 {a}).coprod (𝓟 {i})) =
      𝓟 (({b} : Set β) ×ˢ (univ : Set ι)) := by
  /-
    α : Type u_6
    β : Type u_7
    ι : Type u_8
    a : α
    b : β
    i : ι
    ⊢ Eq (Filter.map (Prod.map (fun x => b) id) ((Filter.principal (Singleton.sing …
  -/
  rw [principal_coprod_principal, map_principal]
  /-
    α : Type u_6
    β : Type u_7
    ι : Type u_8
    a : α
    b : β
    i : ι
    ⊢ Eq (Filter.principal (Set.image (Prod.map (fun x => b) id) (HasCompl.compl ( …
  -/
  congr
  /-
    case e_s
    α : Type u_6
    β : Type u_7
    ι : Type u_8
    a : α
    b : β
    i : ι
    ⊢ Eq (Set.image (Prod.map (fun x => b) id) (HasCompl.compl (SProd.sprod (HasCo …
  -/
  ext ⟨b', i'⟩
  /-
    case e_s.h.mk
    α : Type u_6
    β : Type u_7
    ι : Type u_8
    a : α
    b : β
    i : ι
    b' : β
    i' : ι
    ⊢ Iff (Membership.mem (Set.image (Prod.map (fun x => b) id) (HasCompl.compl (S …
  -/
  constructor
    /-
      case e_s.h.mk.mp
      α : Type u_6
      β : Type u_7
      ι : Type u_8
      a : α
      b : β
      i : ι
      b' : β
      i' : ι
      ⊢ Membership.mem (Set.image (Prod.map (fun x => b) id) (HasCompl.compl (SProd. …
    -/
  · rintro ⟨⟨a'', i''⟩, _, h₂, h₃⟩
    /-
      case e_s.h.mk.mp.intro.mk.intro.refl
      α : Type u_6
      β : Type u_7
      ι : Type u_8
      a : α
      b : β
      i : ι
      a'' : α
      i'' : ι
      left✝ : Membership.mem (HasCompl.compl (SProd.sprod (HasCompl.compl (Singleton …
      ⊢ Membership.mem (SProd.sprod (Singleton.singleton b) Set.univ) { fst := (fun  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.mk.mpr
      α : Type u_6
      β : Type u_7
      ι : Type u_8
      a : α
      b : β
      i : ι
      b' : β
      i' : ι
      ⊢ Membership.mem (SProd.sprod (Singleton.singleton b) Set.univ) { fst := b', s …
    -/
  · rintro ⟨h₁, _⟩
    /-
      case e_s.h.mk.mpr.intro
      α : Type u_6
      β : Type u_7
      ι : Type u_8
      a : α
      b : β
      i : ι
      b' : β
      i' : ι
      h₁ : Membership.mem (Singleton.singleton b) { fst := b', snd := i' }.1
      right✝ : Membership.mem Set.univ { fst := b', snd := i' }.2
      ⊢ Membership.mem (Set.image (Prod.map (fun x => b) id) (HasCompl.compl (SProd. …
    -/
    use (a, i')
    /-
      case h
      α : Type u_6
      β : Type u_7
      ι : Type u_8
      a : α
      b : β
      i : ι
      b' : β
      i' : ι
      h₁ : Membership.mem (Singleton.singleton b) { fst := b', snd := i' }.1
      right✝ : Membership.mem Set.univ { fst := b', snd := i' }.2
      ⊢ And (Membership.mem (HasCompl.compl (SProd.sprod (HasCompl.compl (Singleton. …
    -/
    simpa using h₁.symm
    /-
      🎉 no goals
    -/


theorem Tendsto.prod_map_coprod {δ : Type*} {f : α → γ} {g : β → δ} {a : Filter α} {b : Filter β}
    {c : Filter γ} {d : Filter δ} (hf : Tendsto f a c) (hg : Tendsto g b d) :
    Tendsto (Prod.map f g) (a.coprod b) (c.coprod d) :=
  map_prod_map_coprod_le.trans (coprod_mono hf hg)


