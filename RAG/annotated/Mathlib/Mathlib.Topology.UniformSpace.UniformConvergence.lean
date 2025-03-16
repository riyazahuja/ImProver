/-- A sequence of functions `Fₙ` converges uniformly on a filter `p'` to a limiting function `f`
with respect to the filter `p` if, for any entourage of the diagonal `u`, one has
`p ×ˢ p'`-eventually `(f x, Fₙ x) ∈ u`. -/
def TendstoUniformlyOnFilter (F : ι → α → β) (f : α → β) (p : Filter ι) (p' : Filter α) :=
  ∀ u ∈ 𝓤 β, ∀ᶠ n : ι × α in p ×ˢ p', (f n.snd, F n.fst n.snd) ∈ u


/--
A sequence of functions `Fₙ` converges uniformly on a filter `p'` to a limiting function `f` w.r.t.
filter `p` iff the function `(n, x) ↦ (f x, Fₙ x)` converges along `p ×ˢ p'` to the uniformity.
In other words: one knows nothing about the behavior of `x` in this limit besides it being in `p'`.
-/
theorem tendstoUniformlyOnFilter_iff_tendsto :
    TendstoUniformlyOnFilter F f p p' ↔
      Tendsto (fun q : ι × α => (f q.2, F q.1 q.2)) (p ×ˢ p') (𝓤 β) :=
  Iff.rfl


/-- A sequence of functions `Fₙ` converges uniformly on a set `s` to a limiting function `f` with
respect to the filter `p` if, for any entourage of the diagonal `u`, one has `p`-eventually
`(f x, Fₙ x) ∈ u` for all `x ∈ s`. -/
def TendstoUniformlyOn (F : ι → α → β) (f : α → β) (p : Filter ι) (s : Set α) :=
  ∀ u ∈ 𝓤 β, ∀ᶠ n in p, ∀ x : α, x ∈ s → (f x, F n x) ∈ u


theorem tendstoUniformlyOn_iff_tendstoUniformlyOnFilter :
    TendstoUniformlyOn F f p s ↔ TendstoUniformlyOnFilter F f p (𝓟 s) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    ⊢ Iff (TendstoUniformlyOn F f p s) (TendstoUniformlyOnFilter F f p (Filter.pri …
  -/
  simp only [TendstoUniformlyOn, TendstoUniformlyOnFilter]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    ⊢ Iff (∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Filter.Eventu …
  -/
  apply forall₂_congr
  /-
    case h
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    ⊢ ∀ (a : Set (Prod β β)), Membership.mem (uniformity β) a → Iff (Filter.Eventu …
  -/
  simp_rw [eventually_prod_principal_iff]
  /-
    case h
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    ⊢ ∀ (a : Set (Prod β β)), Membership.mem (uniformity β) a → True
  -/
  simp
  /-
    🎉 no goals
  -/



alias ⟨TendstoUniformlyOn.tendstoUniformlyOnFilter, TendstoUniformlyOnFilter.tendstoUniformlyOn⟩ :=
  tendstoUniformlyOn_iff_tendstoUniformlyOnFilter


/-- A sequence of functions `Fₙ` converges uniformly on a set `s` to a limiting function `f` w.r.t.
filter `p` iff the function `(n, x) ↦ (f x, Fₙ x)` converges along `p ×ˢ 𝓟 s` to the uniformity.
In other words: one knows nothing about the behavior of `x` in this limit besides it being in `s`.
-/
theorem tendstoUniformlyOn_iff_tendsto {F : ι → α → β} {f : α → β} {p : Filter ι} {s : Set α} :
    TendstoUniformlyOn F f p s ↔
    Tendsto (fun q : ι × α => (f q.2, F q.1 q.2)) (p ×ˢ 𝓟 s) (𝓤 β) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    s : Set α
    ⊢ Iff (TendstoUniformlyOn F f p s) (Filter.Tendsto (fun q => { fst := f q.2, s …
  -/
  simp [tendstoUniformlyOn_iff_tendstoUniformlyOnFilter, tendstoUniformlyOnFilter_iff_tendsto]
  /-
    🎉 no goals
  -/


/-- A sequence of functions `Fₙ` converges uniformly to a limiting function `f` with respect to a
filter `p` if, for any entourage of the diagonal `u`, one has `p`-eventually
`(f x, Fₙ x) ∈ u` for all `x`. -/
def TendstoUniformly (F : ι → α → β) (f : α → β) (p : Filter ι) :=
  ∀ u ∈ 𝓤 β, ∀ᶠ n in p, ∀ x : α, (f x, F n x) ∈ u

-- Porting note: moved from below

theorem tendstoUniformlyOn_univ : TendstoUniformlyOn F f p univ ↔ TendstoUniformly F f p := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    ⊢ Iff (TendstoUniformlyOn F f p Set.univ) (TendstoUniformly F f p)
  -/
  simp [TendstoUniformlyOn, TendstoUniformly]
  /-
    🎉 no goals
  -/


theorem tendstoUniformly_iff_tendstoUniformlyOnFilter :
    TendstoUniformly F f p ↔ TendstoUniformlyOnFilter F f p ⊤ := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    ⊢ Iff (TendstoUniformly F f p) (TendstoUniformlyOnFilter F f p Top.top)
  -/
  rw [← tendstoUniformlyOn_univ, tendstoUniformlyOn_iff_tendstoUniformlyOnFilter, principal_univ]
  /-
    🎉 no goals
  -/


theorem TendstoUniformly.tendstoUniformlyOnFilter (h : TendstoUniformly F f p) :
                                           /-
                                             α : Type u
                                             β : Type v
                                             ι : Type x
                                             inst✝ : UniformSpace β
                                             F : ι → α → β
                                             f : α → β
                                             p : Filter ι
                                             h : TendstoUniformly F f p
                                             ⊢ TendstoUniformlyOnFilter F f p Top.top
                                           -/
    TendstoUniformlyOnFilter F f p ⊤ := by rwa [← tendstoUniformly_iff_tendstoUniformlyOnFilter]
                                           /-
                                             🎉 no goals
                                           -/


theorem tendstoUniformlyOn_iff_tendstoUniformly_comp_coe :
    TendstoUniformlyOn F f p s ↔ TendstoUniformly (fun i (x : s) => F i x) (f ∘ (↑)) p :=
                              /-
                                α : Type u
                                β : Type v
                                ι : Type x
                                inst✝ : UniformSpace β
                                F : ι → α → β
                                f : α → β
                                s : Set α
                                p : Filter ι
                                u : Set (Prod β β)
                                x✝ : Membership.mem (uniformity β) u
                                ⊢ Iff (Filter.Eventually (fun n => ∀ (x : α), Membership.mem s x → Membership. …
                              -/
  forall₂_congr fun u _ => by simp
                              /-
                                🎉 no goals
                              -/


/-- A sequence of functions `Fₙ` converges uniformly to a limiting function `f` w.r.t.
filter `p` iff the function `(n, x) ↦ (f x, Fₙ x)` converges along `p ×ˢ ⊤` to the uniformity.
In other words: one knows nothing about the behavior of `x` in this limit.
-/
theorem tendstoUniformly_iff_tendsto {F : ι → α → β} {f : α → β} {p : Filter ι} :
    TendstoUniformly F f p ↔ Tendsto (fun q : ι × α => (f q.2, F q.1 q.2)) (p ×ˢ ⊤) (𝓤 β) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    ⊢ Iff (TendstoUniformly F f p) (Filter.Tendsto (fun q => { fst := f q.2, snd : …
  -/
  simp [tendstoUniformly_iff_tendstoUniformlyOnFilter, tendstoUniformlyOnFilter_iff_tendsto]
  /-
    🎉 no goals
  -/


/-- Uniform converence implies pointwise convergence. -/
theorem TendstoUniformlyOnFilter.tendsto_at (h : TendstoUniformlyOnFilter F f p p')
    (hx : 𝓟 {x} ≤ p') : Tendsto (fun n => F n x) p <| 𝓝 (f x) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    p' : Filter α
    h : TendstoUniformlyOnFilter F f p p'
    hx : LE.le (Filter.principal (Singleton.singleton x)) p'
    ⊢ Filter.Tendsto (fun n => F n x) p (nhds (f x))
  -/
  refine Uniform.tendsto_nhds_right.mpr fun u hu => mem_map.mpr ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    p' : Filter α
    h : TendstoUniformlyOnFilter F f p p'
    hx : LE.le (Filter.principal (Singleton.singleton x)) p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Membership.mem p (Set.preimage (fun x_1 => { fst := f x, snd := F x_1 x }) u)
  -/
  filter_upwards [(h u hu).curry]
  /-
    case h
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    p' : Filter α
    h : TendstoUniformlyOnFilter F f p p'
    hx : LE.le (Filter.principal (Singleton.singleton x)) p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ ∀ (a : ι), Filter.Eventually (fun y => Membership.mem u { fst := f y, snd := …
  -/
  intro i h
  /-
    case h
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    p' : Filter α
    h✝ : TendstoUniformlyOnFilter F f p p'
    hx : LE.le (Filter.principal (Singleton.singleton x)) p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    i : ι
    h : Filter.Eventually (fun y => Membership.mem u { fst := f y, snd := F i y }) …
    ⊢ Membership.mem (Set.preimage (fun x_1 => { fst := f x, snd := F x_1 x }) u) i
  -/
  simpa using h.filter_mono hx
  /-
    🎉 no goals
  -/


/-- Uniform converence implies pointwise convergence. -/
theorem TendstoUniformlyOn.tendsto_at (h : TendstoUniformlyOn F f p s) {x : α} (hx : x ∈ s) :
    Tendsto (fun n => F n x) p <| 𝓝 (f x) :=
  h.tendstoUniformlyOnFilter.tendsto_at
    (le_principal_iff.mpr <| mem_principal.mpr <| singleton_subset_iff.mpr <| hx)


/-- Uniform converence implies pointwise convergence. -/
theorem TendstoUniformly.tendsto_at (h : TendstoUniformly F f p) (x : α) :
    Tendsto (fun n => F n x) p <| 𝓝 (f x) :=
  h.tendstoUniformlyOnFilter.tendsto_at le_top

-- Porting note: tendstoUniformlyOn_univ moved up


theorem TendstoUniformlyOnFilter.mono_left {p'' : Filter ι} (h : TendstoUniformlyOnFilter F f p p')
    (hp : p'' ≤ p) : TendstoUniformlyOnFilter F f p'' p' := fun u hu =>
  (h u hu).filter_mono (p'.prod_mono_left hp)


theorem TendstoUniformlyOnFilter.mono_right {p'' : Filter α} (h : TendstoUniformlyOnFilter F f p p')
    (hp : p'' ≤ p') : TendstoUniformlyOnFilter F f p p'' := fun u hu =>
  (h u hu).filter_mono (p.prod_mono_right hp)


theorem TendstoUniformlyOn.mono {s' : Set α} (h : TendstoUniformlyOn F f p s) (h' : s' ⊆ s) :
    TendstoUniformlyOn F f p s' :=
  tendstoUniformlyOn_iff_tendstoUniformlyOnFilter.mpr
    (h.tendstoUniformlyOnFilter.mono_right (le_principal_iff.mpr <| mem_principal.mpr h'))


theorem TendstoUniformlyOnFilter.congr {F' : ι → α → β} (hf : TendstoUniformlyOnFilter F f p p')
    (hff' : ∀ᶠ n : ι × α in p ×ˢ p', F n.fst n.snd = F' n.fst n.snd) :
    TendstoUniformlyOnFilter F' f p p' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p p'
    hff' : Filter.Eventually (fun n => Eq (F n.1 n.2) (F' n.1 n.2)) (SProd.sprod p …
    ⊢ TendstoUniformlyOnFilter F' f p p'
  -/
  refine fun u hu => ((hf u hu).and hff').mono fun n h => ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p p'
    hff' : Filter.Eventually (fun n => Eq (F n.1 n.2) (F' n.1 n.2)) (SProd.sprod p …
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    n : Prod ι α
    h : And (Membership.mem u { fst := f n.2, snd := F n.1 n.2 }) (Eq (F n.1 n.2)  …
    ⊢ Membership.mem u { fst := f n.2, snd := F' n.1 n.2 }
  -/
  rw [← h.right]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p p'
    hff' : Filter.Eventually (fun n => Eq (F n.1 n.2) (F' n.1 n.2)) (SProd.sprod p …
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    n : Prod ι α
    h : And (Membership.mem u { fst := f n.2, snd := F n.1 n.2 }) (Eq (F n.1 n.2)  …
    ⊢ Membership.mem u { fst := f n.2, snd := F n.1 n.2 }
  -/
  exact h.left
  /-
    🎉 no goals
  -/


theorem TendstoUniformlyOn.congr {F' : ι → α → β} (hf : TendstoUniformlyOn F f p s)
    (hff' : ∀ᶠ n in p, Set.EqOn (F n) (F' n) s) : TendstoUniformlyOn F' f p s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    F' : ι → α → β
    hf : TendstoUniformlyOn F f p s
    hff' : Filter.Eventually (fun n => Set.EqOn (F n) (F' n) s) p
    ⊢ TendstoUniformlyOn F' f p s
  -/
  rw [tendstoUniformlyOn_iff_tendstoUniformlyOnFilter] at hf ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p (Filter.principal s)
    hff' : Filter.Eventually (fun n => Set.EqOn (F n) (F' n) s) p
    ⊢ TendstoUniformlyOnFilter F' f p (Filter.principal s)
  -/
  refine hf.congr ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p (Filter.principal s)
    hff' : Filter.Eventually (fun n => Set.EqOn (F n) (F' n) s) p
    ⊢ Filter.Eventually (fun n => Eq (F n.1 n.2) (F' n.1 n.2)) (SProd.sprod p (Fil …
  -/
  rw [eventually_iff] at hff' ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p (Filter.principal s)
    hff' : Membership.mem p (setOf fun x => Set.EqOn (F x) (F' x) s)
    ⊢ Membership.mem (SProd.sprod p (Filter.principal s)) (setOf fun x => Eq (F x. …
  -/
  simp only [Set.EqOn] at hff'
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    F' : ι → α → β
    hf : TendstoUniformlyOnFilter F f p (Filter.principal s)
    hff' : Membership.mem p (setOf fun x => ∀ ⦃x_1 : α⦄, Membership.mem s x_1 → Eq …
    ⊢ Membership.mem (SProd.sprod p (Filter.principal s)) (setOf fun x => Eq (F x. …
  -/
  simp only [mem_prod_principal, hff', mem_setOf_eq]
  /-
    🎉 no goals
  -/


lemma tendstoUniformly_congr {F F' : ι → α → β} {f : α → β} (hF : F =ᶠ[p] F') :
    TendstoUniformly F f p ↔ TendstoUniformly F' f p := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    p : Filter ι
    F F' : ι → α → β
    f : α → β
    hF : p.EventuallyEq F F'
    ⊢ Iff (TendstoUniformly F f p) (TendstoUniformly F' f p)
  -/
  simp_rw [← tendstoUniformlyOn_univ] at *
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    p : Filter ι
    F F' : ι → α → β
    f : α → β
    hF : p.EventuallyEq F F'
    ⊢ Iff (TendstoUniformlyOn F f p Set.univ) (TendstoUniformlyOn F' f p Set.univ)
  -/
  have HF := EventuallyEq.exists_mem hF
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    p : Filter ι
    F F' : ι → α → β
    f : α → β
    hF : p.EventuallyEq F F'
    HF : Exists fun s => And (Membership.mem p s) (Set.EqOn F F' s)
    ⊢ Iff (TendstoUniformlyOn F f p Set.univ) (TendstoUniformlyOn F' f p Set.univ)
  -/
  exact ⟨fun h => h.congr (by aesop), fun h => h.congr (by simp_rw [eqOn_comm]; aesop)⟩
  /-
    🎉 no goals
  -/


theorem TendstoUniformlyOn.congr_right {g : α → β} (hf : TendstoUniformlyOn F f p s)
    (hfg : EqOn f g s) : TendstoUniformlyOn F g p s := fun u hu => by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    g : α → β
    hf : TendstoUniformlyOn F f p s
    hfg : Set.EqOn f g s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem s x → Membership.mem u …
  -/
  filter_upwards [hf u hu] with i hi a ha using hfg ha ▸ hi a ha
  /-
    🎉 no goals
  -/


protected theorem TendstoUniformly.tendstoUniformlyOn (h : TendstoUniformly F f p) :
    TendstoUniformlyOn F f p s :=
  (tendstoUniformlyOn_univ.2 h).mono (subset_univ s)


/-- Composing on the right by a function preserves uniform convergence on a filter -/
theorem TendstoUniformlyOnFilter.comp (h : TendstoUniformlyOnFilter F f p p') (g : γ → α) :
    TendstoUniformlyOnFilter (fun n => F n ∘ g) (f ∘ g) p (p'.comap g) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    h : TendstoUniformlyOnFilter F f p p'
    g : γ → α
    ⊢ TendstoUniformlyOnFilter (fun n => Function.comp (F n) g) (Function.comp f g …
  -/
  rw [tendstoUniformlyOnFilter_iff_tendsto] at h ⊢
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    h : Filter.Tendsto (fun q => { fst := f q.2, snd := F q.1 q.2 }) (SProd.sprod  …
    g : γ → α
    ⊢ Filter.Tendsto (fun q => { fst := Function.comp f g q.2, snd := Function.com …
  -/
  exact h.comp (tendsto_id.prod_map tendsto_comap)
  /-
    🎉 no goals
  -/


/-- Composing on the right by a function preserves uniform convergence on a set -/
theorem TendstoUniformlyOn.comp (h : TendstoUniformlyOn F f p s) (g : γ → α) :
    TendstoUniformlyOn (fun n => F n ∘ g) (f ∘ g) p (g ⁻¹' s) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    h : TendstoUniformlyOn F f p s
    g : γ → α
    ⊢ TendstoUniformlyOn (fun n => Function.comp (F n) g) (Function.comp f g) p (S …
  -/
  rw [tendstoUniformlyOn_iff_tendstoUniformlyOnFilter] at h ⊢
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    h : TendstoUniformlyOnFilter F f p (Filter.principal s)
    g : γ → α
    ⊢ TendstoUniformlyOnFilter (fun n => Function.comp (F n) g) (Function.comp f g …
  -/
  simpa [TendstoUniformlyOn, comap_principal] using TendstoUniformlyOnFilter.comp h g
  /-
    🎉 no goals
  -/


/-- Composing on the right by a function preserves uniform convergence -/
theorem TendstoUniformly.comp (h : TendstoUniformly F f p) (g : γ → α) :
    TendstoUniformly (fun n => F n ∘ g) (f ∘ g) p := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    h : TendstoUniformly F f p
    g : γ → α
    ⊢ TendstoUniformly (fun n => Function.comp (F n) g) (Function.comp f g) p
  -/
  rw [tendstoUniformly_iff_tendstoUniformlyOnFilter] at h ⊢
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    h : TendstoUniformlyOnFilter F f p Top.top
    g : γ → α
    ⊢ TendstoUniformlyOnFilter (fun n => Function.comp (F n) g) (Function.comp f g …
  -/
  simpa [principal_univ, comap_principal] using h.comp g
  /-
    🎉 no goals
  -/


/-- Composing on the left by a uniformly continuous function preserves
  uniform convergence on a filter -/
theorem UniformContinuous.comp_tendstoUniformlyOnFilter [UniformSpace γ] {g : β → γ}
    (hg : UniformContinuous g) (h : TendstoUniformlyOnFilter F f p p') :
    TendstoUniformlyOnFilter (fun i => g ∘ F i) (g ∘ f) p p' := fun _u hu => h _ (hg hu)


/-- Composing on the left by a uniformly continuous function preserves
  uniform convergence on a set -/
theorem UniformContinuous.comp_tendstoUniformlyOn [UniformSpace γ] {g : β → γ}
    (hg : UniformContinuous g) (h : TendstoUniformlyOn F f p s) :
    TendstoUniformlyOn (fun i => g ∘ F i) (g ∘ f) p s := fun _u hu => h _ (hg hu)


/-- Composing on the left by a uniformly continuous function preserves uniform convergence -/
theorem UniformContinuous.comp_tendstoUniformly [UniformSpace γ] {g : β → γ}
    (hg : UniformContinuous g) (h : TendstoUniformly F f p) :
    TendstoUniformly (fun i => g ∘ F i) (g ∘ f) p := fun _u hu => h _ (hg hu)


theorem TendstoUniformlyOnFilter.prod_map {ι' α' β' : Type*} [UniformSpace β'] {F' : ι' → α' → β'}
    {f' : α' → β'} {q : Filter ι'} {q' : Filter α'} (h : TendstoUniformlyOnFilter F f p p')
    (h' : TendstoUniformlyOnFilter F' f' q q') :
    TendstoUniformlyOnFilter (fun i : ι × ι' => Prod.map (F i.1) (F' i.2)) (Prod.map f f')
      (p ×ˢ q) (p' ×ˢ q') := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    q : Filter ι'
    q' : Filter α'
    h : TendstoUniformlyOnFilter F f p p'
    h' : TendstoUniformlyOnFilter F' f' q q'
    ⊢ TendstoUniformlyOnFilter (fun i => Prod.map (F i.1) (F' i.2)) (Prod.map f f' …
  -/
  rw [tendstoUniformlyOnFilter_iff_tendsto] at h h' ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    q : Filter ι'
    q' : Filter α'
    h : Filter.Tendsto (fun q => { fst := f q.2, snd := F q.1 q.2 }) (SProd.sprod  …
    h' : Filter.Tendsto (fun q => { fst := f' q.2, snd := F' q.1 q.2 }) (SProd.spr …
    ⊢ Filter.Tendsto (fun q => { fst := Prod.map f f' q.2, snd := Prod.map (F q.1. …
  -/
  rw [uniformity_prod_eq_comap_prod, tendsto_comap_iff, ← map_swap4_prod, tendsto_map'_iff]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    q : Filter ι'
    q' : Filter α'
    h : Filter.Tendsto (fun q => { fst := f q.2, snd := F q.1 q.2 }) (SProd.sprod  …
    h' : Filter.Tendsto (fun q => { fst := f' q.2, snd := F' q.1 q.2 }) (SProd.spr …
    ⊢ Filter.Tendsto (Function.comp (Function.comp (fun p => { fst := { fst := p.1 …
  -/
  convert h.prod_map h' -- seems to be faster than `exact` here
  /-
    🎉 no goals
  -/


theorem TendstoUniformlyOn.prod_map {ι' α' β' : Type*} [UniformSpace β'] {F' : ι' → α' → β'}
    {f' : α' → β'} {p' : Filter ι'} {s' : Set α'} (h : TendstoUniformlyOn F f p s)
    (h' : TendstoUniformlyOn F' f' p' s') :
    TendstoUniformlyOn (fun i : ι × ι' => Prod.map (F i.1) (F' i.2)) (Prod.map f f') (p ×ˢ p')
      (s ×ˢ s') := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    p' : Filter ι'
    s' : Set α'
    h : TendstoUniformlyOn F f p s
    h' : TendstoUniformlyOn F' f' p' s'
    ⊢ TendstoUniformlyOn (fun i => Prod.map (F i.1) (F' i.2)) (Prod.map f f') (SPr …
  -/
  rw [tendstoUniformlyOn_iff_tendstoUniformlyOnFilter] at h h' ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    p' : Filter ι'
    s' : Set α'
    h : TendstoUniformlyOnFilter F f p (Filter.principal s)
    h' : TendstoUniformlyOnFilter F' f' p' (Filter.principal s')
    ⊢ TendstoUniformlyOnFilter (fun i => Prod.map (F i.1) (F' i.2)) (Prod.map f f' …
  -/
  simpa only [prod_principal_principal] using h.prod_map h'
  /-
    🎉 no goals
  -/


theorem TendstoUniformly.prod_map {ι' α' β' : Type*} [UniformSpace β'] {F' : ι' → α' → β'}
    {f' : α' → β'} {p' : Filter ι'} (h : TendstoUniformly F f p) (h' : TendstoUniformly F' f' p') :
    TendstoUniformly (fun i : ι × ι' => Prod.map (F i.1) (F' i.2)) (Prod.map f f') (p ×ˢ p') := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    p' : Filter ι'
    h : TendstoUniformly F f p
    h' : TendstoUniformly F' f' p'
    ⊢ TendstoUniformly (fun i => Prod.map (F i.1) (F' i.2)) (Prod.map f f') (SProd …
  -/
  rw [← tendstoUniformlyOn_univ, ← univ_prod_univ] at *
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    f' : α' → β'
    p' : Filter ι'
    h : TendstoUniformlyOn F f p Set.univ
    h' : TendstoUniformlyOn F' f' p' Set.univ
    ⊢ TendstoUniformlyOn (fun i => Prod.map (F i.1) (F' i.2)) (Prod.map f f') (SPr …
  -/
  exact h.prod_map h'
  /-
    🎉 no goals
  -/


theorem TendstoUniformlyOnFilter.prod {ι' β' : Type*} [UniformSpace β'] {F' : ι' → α → β'}
    {f' : α → β'} {q : Filter ι'} (h : TendstoUniformlyOnFilter F f p p')
    (h' : TendstoUniformlyOnFilter F' f' q p') :
    TendstoUniformlyOnFilter (fun (i : ι × ι') a => (F i.1 a, F' i.2 a)) (fun a => (f a, f' a))
      (p ×ˢ q) p' :=
  fun u hu => ((h.prod_map h') u hu).diag_of_prod_right


protected theorem TendstoUniformlyOn.prod {ι' β' : Type*} [UniformSpace β']
    {F' : ι' → α → β'} {f' : α → β'} {p' : Filter ι'}
    (h : TendstoUniformlyOn F f p s) (h' : TendstoUniformlyOn F' f' p' s) :
    TendstoUniformlyOn (fun (i : ι × ι') a => (F i.1 a, F' i.2 a)) (fun a => (f a, f' a))
      (p ×ˢ p') s :=
  (congr_arg _ s.inter_self).mp ((h.prod_map h').comp fun a => (a, a))


theorem TendstoUniformly.prod {ι' β' : Type*} [UniformSpace β'] {F' : ι' → α → β'} {f' : α → β'}
    {p' : Filter ι'} (h : TendstoUniformly F f p) (h' : TendstoUniformly F' f' p') :
    TendstoUniformly (fun (i : ι × ι') a => (F i.1 a, F' i.2 a)) (fun a => (f a, f' a))
      (p ×ˢ p') :=
  (h.prod_map h').comp fun a => (a, a)


/-- Uniform convergence on a filter `p'` to a constant function is equivalent to convergence in
`p ×ˢ p'`. -/
theorem tendsto_prod_filter_iff {c : β} :
    Tendsto (↿F) (p ×ˢ p') (𝓝 c) ↔ TendstoUniformlyOnFilter F (fun _ => c) p p' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    c : β
    ⊢ Iff (Filter.Tendsto (Function.HasUncurry.uncurry F) (SProd.sprod p p') (nhds …
  -/
  simp_rw [nhds_eq_comap_uniformity, tendsto_comap_iff]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    c : β
    ⊢ Iff (Filter.Tendsto (Function.comp (Prod.mk c) (Function.HasUncurry.uncurry  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Uniform convergence on a set `s` to a constant function is equivalent to convergence in
`p ×ˢ 𝓟 s`. -/
theorem tendsto_prod_principal_iff {c : β} :
    Tendsto (↿F) (p ×ˢ 𝓟 s) (𝓝 c) ↔ TendstoUniformlyOn F (fun _ => c) p s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    c : β
    ⊢ Iff (Filter.Tendsto (Function.HasUncurry.uncurry F) (SProd.sprod p (Filter.p …
  -/
  rw [tendstoUniformlyOn_iff_tendstoUniformlyOnFilter]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    c : β
    ⊢ Iff (Filter.Tendsto (Function.HasUncurry.uncurry F) (SProd.sprod p (Filter.p …
  -/
  exact tendsto_prod_filter_iff
  /-
    🎉 no goals
  -/


/-- Uniform convergence to a constant function is equivalent to convergence in `p ×ˢ ⊤`. -/
theorem tendsto_prod_top_iff {c : β} :
    Tendsto (↿F) (p ×ˢ ⊤) (𝓝 c) ↔ TendstoUniformly F (fun _ => c) p := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    c : β
    ⊢ Iff (Filter.Tendsto (Function.HasUncurry.uncurry F) (SProd.sprod p Top.top)  …
  -/
  rw [tendstoUniformly_iff_tendstoUniformlyOnFilter]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    c : β
    ⊢ Iff (Filter.Tendsto (Function.HasUncurry.uncurry F) (SProd.sprod p Top.top)  …
  -/
  exact tendsto_prod_filter_iff
  /-
    🎉 no goals
  -/


/-- Uniform convergence on the empty set is vacuously true -/
                                                                               /-
                                                                                 α : Type u
                                                                                 β : Type v
                                                                                 ι : Type x
                                                                                 inst✝ : UniformSpace β
                                                                                 F : ι → α → β
                                                                                 f : α → β
                                                                                 p : Filter ι
                                                                                 u : Set (Prod β β)
                                                                                 x✝ : Membership.mem (uniformity β) u
                                                                                 ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem EmptyCollection.emptyC …
                                                                               -/
theorem tendstoUniformlyOn_empty : TendstoUniformlyOn F f p ∅ := fun u _ => by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- Uniform convergence on a singleton is equivalent to regular convergence -/
theorem tendstoUniformlyOn_singleton_iff_tendsto :
    TendstoUniformlyOn F f p {x} ↔ Tendsto (fun n : ι => F n x) p (𝓝 (f x)) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    ⊢ Iff (TendstoUniformlyOn F f p (Singleton.singleton x)) (Filter.Tendsto (fun  …
  -/
  simp_rw [tendstoUniformlyOn_iff_tendsto, Uniform.tendsto_nhds_right, tendsto_def]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    ⊢ Iff (∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Membership.me …
  -/
  exact forall₂_congr fun u _ => by simp [mem_prod_principal, preimage]
  /-
    🎉 no goals
  -/


/-- If a sequence `g` converges to some `b`, then the sequence of constant functions
`fun n ↦ fun a ↦ g n` converges to the constant function `fun a ↦ b` on any set `s` -/
theorem Filter.Tendsto.tendstoUniformlyOnFilter_const {g : ι → β} {b : β} (hg : Tendsto g p (𝓝 b))
    (p' : Filter α) :
    TendstoUniformlyOnFilter (fun n : ι => fun _ : α => g n) (fun _ : α => b) p p' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    p : Filter ι
    g : ι → β
    b : β
    hg : Filter.Tendsto g p (nhds b)
    p' : Filter α
    ⊢ TendstoUniformlyOnFilter (fun n x => g n) (fun x => b) p p'
  -/
  simpa only [nhds_eq_comap_uniformity, tendsto_comap_iff] using hg.comp (tendsto_fst (g := p'))
  /-
    🎉 no goals
  -/


/-- If a sequence `g` converges to some `b`, then the sequence of constant functions
`fun n ↦ fun a ↦ g n` converges to the constant function `fun a ↦ b` on any set `s` -/
theorem Filter.Tendsto.tendstoUniformlyOn_const {g : ι → β} {b : β} (hg : Tendsto g p (𝓝 b))
    (s : Set α) : TendstoUniformlyOn (fun n : ι => fun _ : α => g n) (fun _ : α => b) p s :=
  tendstoUniformlyOn_iff_tendstoUniformlyOnFilter.mpr (hg.tendstoUniformlyOnFilter_const (𝓟 s))


theorem UniformContinuousOn.tendstoUniformlyOn [UniformSpace α] [UniformSpace γ] {x : α} {U : Set α}
    {V : Set β} {F : α → β → γ} (hF : UniformContinuousOn (↿F) (U ×ˢ V)) (hU : x ∈ U) :
    TendstoUniformlyOn F (F x) (𝓝[U] x) V := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace γ
    x : α
    U : Set α
    V : Set β
    F : α → β → γ
    hF : UniformContinuousOn (Function.HasUncurry.uncurry F) (SProd.sprod U V)
    hU : Membership.mem U x
    ⊢ TendstoUniformlyOn F (F x) (nhdsWithin x U) V
  -/
  set φ := fun q : α × β => ((x, q.2), q)
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace γ
    x : α
    U : Set α
    V : Set β
    F : α → β → γ
    hF : UniformContinuousOn (Function.HasUncurry.uncurry F) (SProd.sprod U V)
    hU : Membership.mem U x
    φ : Prod α β → Prod (Prod α β) (Prod α β) := fun q => { fst := { fst := x, snd …
    ⊢ TendstoUniformlyOn F (F x) (nhdsWithin x U) V
  -/
  rw [tendstoUniformlyOn_iff_tendsto]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace γ
    x : α
    U : Set α
    V : Set β
    F : α → β → γ
    hF : UniformContinuousOn (Function.HasUncurry.uncurry F) (SProd.sprod U V)
    hU : Membership.mem U x
    φ : Prod α β → Prod (Prod α β) (Prod α β) := fun q => { fst := { fst := x, snd …
    ⊢ Filter.Tendsto (fun q => { fst := F x q.2, snd := F q.1 q.2 }) (SProd.sprod  …
  -/
  change Tendsto (Prod.map (↿F) ↿F ∘ φ) (𝓝[U] x ×ˢ 𝓟 V) (𝓤 γ)
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace γ
    x : α
    U : Set α
    V : Set β
    F : α → β → γ
    hF : UniformContinuousOn (Function.HasUncurry.uncurry F) (SProd.sprod U V)
    hU : Membership.mem U x
    φ : Prod α β → Prod (Prod α β) (Prod α β) := fun q => { fst := { fst := x, snd …
    ⊢ Filter.Tendsto (Function.comp (Prod.map (Function.HasUncurry.uncurry F) (Fun …
  -/
  simp only [nhdsWithin, Filter.prod_eq_inf, comap_inf, inf_assoc, comap_principal, inf_principal]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace γ
    x : α
    U : Set α
    V : Set β
    F : α → β → γ
    hF : UniformContinuousOn (Function.HasUncurry.uncurry F) (SProd.sprod U V)
    hU : Membership.mem U x
    φ : Prod α β → Prod (Prod α β) (Prod α β) := fun q => { fst := { fst := x, snd …
    ⊢ Filter.Tendsto (Function.comp (Prod.map (Function.HasUncurry.uncurry F) (Fun …
  -/
  refine hF.comp (Tendsto.inf ?_ <| tendsto_principal_principal.2 fun x hx => ⟨⟨hU, hx.2⟩, hx⟩)
  simp only [uniformity_prod_eq_comap_prod, tendsto_comap_iff, (· ∘ ·),
    nhds_eq_comap_uniformity, comap_comap]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace γ
    x : α
    U : Set α
    V : Set β
    F : α → β → γ
    hF : UniformContinuousOn (Function.HasUncurry.uncurry F) (SProd.sprod U V)
    hU : Membership.mem U x
    φ : Prod α β → Prod (Prod α β) (Prod α β) := fun q => { fst := { fst := x, snd …
    ⊢ Filter.Tendsto (Function.comp (fun p => { fst := { fst := p.1.1, snd := p.2. …
  -/
  exact tendsto_comap.prod_mk (tendsto_diag_uniformity _ _)
  /-
    🎉 no goals
  -/


theorem UniformContinuousOn.tendstoUniformly [UniformSpace α] [UniformSpace γ] {x : α} {U : Set α}
    (hU : U ∈ 𝓝 x) {F : α → β → γ} (hF : UniformContinuousOn (↿F) (U ×ˢ (univ : Set β))) :
    TendstoUniformly F (F x) (𝓝 x) := by
  simpa only [tendstoUniformlyOn_univ, nhdsWithin_eq_nhds.2 hU]
    using hF.tendstoUniformlyOn (mem_of_mem_nhds hU)


theorem UniformContinuous₂.tendstoUniformly [UniformSpace α] [UniformSpace γ] {f : α → β → γ}
    (h : UniformContinuous₂ f) {x : α} : TendstoUniformly f (f x) (𝓝 x) :=
                                                      /-
                                                        α : Type u
                                                        β : Type v
                                                        γ : Type w
                                                        inst✝² : UniformSpace β
                                                        inst✝¹ : UniformSpace α
                                                        inst✝ : UniformSpace γ
                                                        f : α → β → γ
                                                        h : UniformContinuous₂ f
                                                        x : α
                                                        ⊢ UniformContinuousOn (Function.HasUncurry.uncurry f) (SProd.sprod Set.univ Se …
                                                      -/
  UniformContinuousOn.tendstoUniformly univ_mem <| by rwa [univ_prod_univ, uniformContinuousOn_univ]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- A sequence is uniformly Cauchy if eventually all of its pairwise differences are
uniformly bounded -/
def UniformCauchySeqOnFilter (F : ι → α → β) (p : Filter ι) (p' : Filter α) : Prop :=
  ∀ u ∈ 𝓤 β, ∀ᶠ m : (ι × ι) × α in (p ×ˢ p) ×ˢ p', (F m.fst.fst m.snd, F m.fst.snd m.snd) ∈ u


/-- A sequence is uniformly Cauchy if eventually all of its pairwise differences are
uniformly bounded -/
def UniformCauchySeqOn (F : ι → α → β) (p : Filter ι) (s : Set α) : Prop :=
  ∀ u ∈ 𝓤 β, ∀ᶠ m : ι × ι in p ×ˢ p, ∀ x : α, x ∈ s → (F m.fst x, F m.snd x) ∈ u


theorem uniformCauchySeqOn_iff_uniformCauchySeqOnFilter :
    UniformCauchySeqOn F p s ↔ UniformCauchySeqOnFilter F p (𝓟 s) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ⊢ Iff (UniformCauchySeqOn F p s) (UniformCauchySeqOnFilter F p (Filter.princip …
  -/
  simp only [UniformCauchySeqOn, UniformCauchySeqOnFilter]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ⊢ Iff (∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Filter.Eventu …
  -/
  refine forall₂_congr fun u hu => ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Iff (Filter.Eventually (fun m => ∀ (x : α), Membership.mem s x → Membership. …
  -/
  rw [eventually_prod_principal_iff]
  /-
    🎉 no goals
  -/


theorem UniformCauchySeqOn.uniformCauchySeqOnFilter (hF : UniformCauchySeqOn F p s) :
                                             /-
                                               α : Type u
                                               β : Type v
                                               ι : Type x
                                               inst✝ : UniformSpace β
                                               F : ι → α → β
                                               s : Set α
                                               p : Filter ι
                                               hF : UniformCauchySeqOn F p s
                                               ⊢ UniformCauchySeqOnFilter F p (Filter.principal s)
                                             -/
    UniformCauchySeqOnFilter F p (𝓟 s) := by rwa [← uniformCauchySeqOn_iff_uniformCauchySeqOnFilter]
                                             /-
                                               🎉 no goals
                                             -/


/-- A sequence that converges uniformly is also uniformly Cauchy -/
theorem TendstoUniformlyOnFilter.uniformCauchySeqOnFilter (hF : TendstoUniformlyOnFilter F f p p') :
    UniformCauchySeqOnFilter F p p' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    ⊢ UniformCauchySeqOnFilter F p p'
  -/
  intro u hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := F m.1.1 m.2, snd := F  …
  -/
  rcases comp_symm_of_uniformity hu with ⟨t, ht, htsymm, htmem⟩
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := F m.1.1 m.2, snd := F  …
  -/
  have := tendsto_swap4_prod.eventually ((hF t ht).prod_mk (hF t ht))
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := f { fst := { …
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := F m.1.1 m.2, snd := F  …
  -/
  apply this.diag_of_prod_right.mono
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := f { fst := { …
    ⊢ ∀ (x : Prod (Prod ι ι) α), And (Membership.mem t { fst := f { fst := { fst : …
  -/
  simp only [and_imp, Prod.forall]
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := f { fst := { …
    ⊢ ∀ (a b : ι) (b_1 : α), Membership.mem t { fst := f b_1, snd := F a b_1 } → M …
  -/
  intro n1 n2 x hl hr
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : TendstoUniformlyOnFilter F f p p'
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := f { fst := { …
    n1 n2 : ι
    x : α
    hl : Membership.mem t { fst := f x, snd := F n1 x }
    hr : Membership.mem t { fst := f x, snd := F n2 x }
    ⊢ Membership.mem u { fst := F n1 x, snd := F n2 x }
  -/
  exact Set.mem_of_mem_of_subset (prod_mk_mem_compRel (htsymm hl) hr) htmem
  /-
    🎉 no goals
  -/


/-- A sequence that converges uniformly is also uniformly Cauchy -/
theorem TendstoUniformlyOn.uniformCauchySeqOn (hF : TendstoUniformlyOn F f p s) :
    UniformCauchySeqOn F p s :=
  uniformCauchySeqOn_iff_uniformCauchySeqOnFilter.mpr
    hF.tendstoUniformlyOnFilter.uniformCauchySeqOnFilter


/-- A uniformly Cauchy sequence converges uniformly to its limit -/
theorem UniformCauchySeqOnFilter.tendstoUniformlyOnFilter_of_tendsto
    (hF : UniformCauchySeqOnFilter F p p')
    (hF' : ∀ᶠ x : α in p', Tendsto (fun n => F n x) p (𝓝 (f x))) :
    TendstoUniformlyOnFilter F f p p' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    ⊢ TendstoUniformlyOnFilter F f p p'
  -/
  rcases p.eq_or_neBot with rfl | _
    /-
      case inl
      α : Type u
      β : Type v
      ι : Type x
      inst✝ : UniformSpace β
      F : ι → α → β
      f : α → β
      p' : Filter α
      hF : UniformCauchySeqOnFilter F Bot.bot p'
      hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) Bot.bot (nhd …
      ⊢ TendstoUniformlyOnFilter F f Bot.bot p'
    -/
  · simp only [TendstoUniformlyOnFilter, bot_prod, eventually_bot, implies_true]
    /-
      🎉 no goals
    -/
  -- Proof idea: |f_n(x) - f(x)| ≤ |f_n(x) - f_m(x)| + |f_m(x) - f(x)|. We choose `n`
  -- so that |f_n(x) - f_m(x)| is uniformly small across `s` whenever `m ≥ n`. Then for
  -- a fixed `x`, we choose `m` sufficiently large such that |f_m(x) - f(x)| is small.
  /-
    case inr
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    ⊢ TendstoUniformlyOnFilter F f p p'
  -/
  intro u hu
  /-
    case inr
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Filter.Eventually (fun n => Membership.mem u { fst := f n.2, snd := F n.1 n. …
  -/
  rcases comp_symm_of_uniformity hu with ⟨t, ht, htsymm, htmem⟩
  -- We will choose n, x, and m simultaneously. n and x come from hF. m comes from hF'
  -- But we need to promote hF' to the full product filter to use it
  have hmc : ∀ᶠ x in (p ×ˢ p) ×ˢ p', Tendsto (fun n : ι => F n x.snd) p (𝓝 (f x.snd)) := by
    rw [eventually_prod_iff]
    exact ⟨fun _ => True, by simp, _, hF', by simp⟩
  -- To apply filter operations we'll need to do some order manipulation
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    ⊢ Filter.Eventually (fun n => Membership.mem u { fst := f n.2, snd := F n.1 n. …
  -/
  rw [Filter.eventually_swap_iff]
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    ⊢ Filter.Eventually (fun y => Membership.mem u { fst := f y.swap.2, snd := F y …
  -/
  have := tendsto_prodAssoc.eventually (tendsto_prod_swap.eventually ((hF t ht).and hmc))
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.pr …
    ⊢ Filter.Eventually (fun y => Membership.mem u { fst := f y.swap.2, snd := F y …
  -/
  apply this.curry.mono
  simp only [Equiv.prodAssoc_apply, eventually_and, eventually_const, Prod.snd_swap, Prod.fst_swap,
    and_imp, Prod.forall]
  -- Complete the proof
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.pr …
    ⊢ ∀ (a : α) (b : ι), Filter.Eventually (fun x => Membership.mem t { fst := F b …
  -/
  intro x n hx hm'
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.pr …
    x : α
    n : ι
    hx : Filter.Eventually (fun x_1 => Membership.mem t { fst := F n x, snd := F x …
    hm' : Filter.Tendsto (fun n => F n x) p (nhds (f x))
    ⊢ Membership.mem u { fst := f x, snd := F n x }
  -/
  refine Set.mem_of_mem_of_subset (mem_compRel.mpr ?_) htmem
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.pr …
    x : α
    n : ι
    hx : Filter.Eventually (fun x_1 => Membership.mem t { fst := F n x, snd := F x …
    hm' : Filter.Tendsto (fun n => F n x) p (nhds (f x))
    ⊢ Exists fun z => And (Membership.mem t { fst := f x, snd := z }) (Membership. …
  -/
  rw [Uniform.tendsto_nhds_right] at hm'
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.pr …
    x : α
    n : ι
    hx : Filter.Eventually (fun x_1 => Membership.mem t { fst := F n x, snd := F x …
    hm' : Filter.Tendsto (fun x_1 => { fst := f x, snd := F x_1 x }) p (uniformity …
    ⊢ Exists fun z => And (Membership.mem t { fst := f x, snd := z }) (Membership. …
  -/
  have := hx.and (hm' ht)
  /-
    case inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this✝ : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.p …
    x : α
    n : ι
    hx : Filter.Eventually (fun x_1 => Membership.mem t { fst := F n x, snd := F x …
    hm' : Filter.Tendsto (fun x_1 => { fst := f x, snd := F x_1 x }) p (uniformity …
    this : Filter.Eventually (fun x_1 => And (Membership.mem t { fst := F n x, snd …
    ⊢ Exists fun z => And (Membership.mem t { fst := f x, snd := z }) (Membership. …
  -/
  obtain ⟨m, hm⟩ := this.exists
  /-
    case inr.intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    hF : UniformCauchySeqOnFilter F p p'
    hF' : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x) p (nhds (f x …
    h✝ : p.NeBot
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    htsymm : ∀ {a b : β}, Membership.mem t { fst := a, snd := b } → Membership.mem …
    htmem : HasSubset.Subset (compRel t t) u
    hmc : Filter.Eventually (fun x => Filter.Tendsto (fun n => F n x.2) p (nhds (f …
    this✝ : Filter.Eventually (fun x => And (Membership.mem t { fst := F ((Equiv.p …
    x : α
    n : ι
    hx : Filter.Eventually (fun x_1 => Membership.mem t { fst := F n x, snd := F x …
    hm' : Filter.Tendsto (fun x_1 => { fst := f x, snd := F x_1 x }) p (uniformity …
    this : Filter.Eventually (fun x_1 => And (Membership.mem t { fst := F n x, snd …
    m : ι
    hm : And (Membership.mem t { fst := F n x, snd := F m x }) (Membership.mem t ( …
    ⊢ Exists fun z => And (Membership.mem t { fst := f x, snd := z }) (Membership. …
  -/
  exact ⟨F m x, ⟨hm.2, htsymm hm.1⟩⟩
  /-
    🎉 no goals
  -/


/-- A uniformly Cauchy sequence converges uniformly to its limit -/
theorem UniformCauchySeqOn.tendstoUniformlyOn_of_tendsto (hF : UniformCauchySeqOn F p s)
    (hF' : ∀ x : α, x ∈ s → Tendsto (fun n => F n x) p (𝓝 (f x))) : TendstoUniformlyOn F f p s :=
  tendstoUniformlyOn_iff_tendstoUniformlyOnFilter.mpr
    (hF.uniformCauchySeqOnFilter.tendstoUniformlyOnFilter_of_tendsto hF')


theorem UniformCauchySeqOnFilter.mono_left {p'' : Filter ι} (hf : UniformCauchySeqOnFilter F p p')
    (hp : p'' ≤ p) : UniformCauchySeqOnFilter F p'' p' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    p'' : Filter ι
    hf : UniformCauchySeqOnFilter F p p'
    hp : LE.le p'' p
    ⊢ UniformCauchySeqOnFilter F p'' p'
  -/
  intro u hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    p'' : Filter ι
    hf : UniformCauchySeqOnFilter F p p'
    hp : LE.le p'' p
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := F m.1.1 m.2, snd := F  …
  -/
  have := (hf u hu).filter_mono (p'.prod_mono_left (Filter.prod_mono hp hp))
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    p'' : Filter ι
    hf : UniformCauchySeqOnFilter F p p'
    hp : LE.le p'' p
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    this : Filter.Eventually (fun x => Membership.mem u { fst := F x.1.1 x.2, snd  …
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := F m.1.1 m.2, snd := F  …
  -/
  exact this.mono (by simp)
  /-
    🎉 no goals
  -/


theorem UniformCauchySeqOnFilter.mono_right {p'' : Filter α} (hf : UniformCauchySeqOnFilter F p p')
    (hp : p'' ≤ p') : UniformCauchySeqOnFilter F p p'' := fun u hu =>
  have := (hf u hu).filter_mono ((p ×ˢ p).prod_mono_right hp)
                /-
                  α : Type u
                  β : Type v
                  ι : Type x
                  inst✝ : UniformSpace β
                  F : ι → α → β
                  p : Filter ι
                  p' p'' : Filter α
                  hf : UniformCauchySeqOnFilter F p p'
                  hp : LE.le p'' p'
                  u : Set (Prod β β)
                  hu : Membership.mem (uniformity β) u
                  this : Filter.Eventually (fun x => Membership.mem u { fst := F x.1.1 x.2, snd  …
                  ⊢ ∀ (x : Prod (Prod ι ι) α), Membership.mem u { fst := F x.1.1 x.2, snd := F x …
                -/
  this.mono (by simp)
                /-
                  🎉 no goals
                -/


theorem UniformCauchySeqOn.mono {s' : Set α} (hf : UniformCauchySeqOn F p s) (hss' : s' ⊆ s) :
    UniformCauchySeqOn F p s' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    s' : Set α
    hf : UniformCauchySeqOn F p s
    hss' : HasSubset.Subset s' s
    ⊢ UniformCauchySeqOn F p s'
  -/
  rw [uniformCauchySeqOn_iff_uniformCauchySeqOnFilter] at hf ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    s' : Set α
    hf : UniformCauchySeqOnFilter F p (Filter.principal s)
    hss' : HasSubset.Subset s' s
    ⊢ UniformCauchySeqOnFilter F p (Filter.principal s')
  -/
  exact hf.mono_right (le_principal_iff.mpr <| mem_principal.mpr hss')
  /-
    🎉 no goals
  -/


/-- Composing on the right by a function preserves uniform Cauchy sequences -/
theorem UniformCauchySeqOnFilter.comp {γ : Type*} (hf : UniformCauchySeqOnFilter F p p')
    (g : γ → α) : UniformCauchySeqOnFilter (fun n => F n ∘ g) p (p'.comap g) := fun u hu => by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    γ : Type u_1
    hf : UniformCauchySeqOnFilter F p p'
    g : γ → α
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := (fun n => Function.com …
  -/
  obtain ⟨pa, hpa, pb, hpb, hpapb⟩ := eventually_prod_iff.mp (hf u hu)
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    γ : Type u_1
    hf : UniformCauchySeqOnFilter F p p'
    g : γ → α
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    pa : Prod ι ι → Prop
    hpa : Filter.Eventually (fun x => pa x) (SProd.sprod p p)
    pb : α → Prop
    hpb : Filter.Eventually (fun y => pb y) p'
    hpapb : ∀ {x : Prod ι ι}, pa x → ∀ {y : α}, pb y → Membership.mem u { fst := F …
    ⊢ Filter.Eventually (fun m => Membership.mem u { fst := (fun n => Function.com …
  -/
  rw [eventually_prod_iff]
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    γ : Type u_1
    hf : UniformCauchySeqOnFilter F p p'
    g : γ → α
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    pa : Prod ι ι → Prop
    hpa : Filter.Eventually (fun x => pa x) (SProd.sprod p p)
    pb : α → Prop
    hpb : Filter.Eventually (fun y => pb y) p'
    hpapb : ∀ {x : Prod ι ι}, pa x → ∀ {y : α}, pb y → Membership.mem u { fst := F …
    ⊢ Exists fun pa => And (Filter.Eventually (fun x => pa x) (SProd.sprod p p)) ( …
  -/
  refine ⟨pa, hpa, pb ∘ g, ?_, fun hx _ hy => hpapb hx hy⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    p : Filter ι
    p' : Filter α
    γ : Type u_1
    hf : UniformCauchySeqOnFilter F p p'
    g : γ → α
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    pa : Prod ι ι → Prop
    hpa : Filter.Eventually (fun x => pa x) (SProd.sprod p p)
    pb : α → Prop
    hpb : Filter.Eventually (fun y => pb y) p'
    hpapb : ∀ {x : Prod ι ι}, pa x → ∀ {y : α}, pb y → Membership.mem u { fst := F …
    ⊢ Filter.Eventually (fun y => Function.comp pb g y) (Filter.comap g p')
  -/
  exact eventually_comap.mpr (hpb.mono fun x hx y hy => by simp only [hx, hy, Function.comp_apply])
  /-
    🎉 no goals
  -/


/-- Composing on the right by a function preserves uniform Cauchy sequences -/
theorem UniformCauchySeqOn.comp {γ : Type*} (hf : UniformCauchySeqOn F p s) (g : γ → α) :
    UniformCauchySeqOn (fun n => F n ∘ g) p (g ⁻¹' s) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    γ : Type u_1
    hf : UniformCauchySeqOn F p s
    g : γ → α
    ⊢ UniformCauchySeqOn (fun n => Function.comp (F n) g) p (Set.preimage g s)
  -/
  rw [uniformCauchySeqOn_iff_uniformCauchySeqOnFilter] at hf ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    γ : Type u_1
    hf : UniformCauchySeqOnFilter F p (Filter.principal s)
    g : γ → α
    ⊢ UniformCauchySeqOnFilter (fun n => Function.comp (F n) g) p (Filter.principa …
  -/
  simpa only [UniformCauchySeqOn, comap_principal] using hf.comp g
  /-
    🎉 no goals
  -/


/-- Composing on the left by a uniformly continuous function preserves
uniform Cauchy sequences -/
theorem UniformContinuous.comp_uniformCauchySeqOn [UniformSpace γ] {g : β → γ}
    (hg : UniformContinuous g) (hf : UniformCauchySeqOn F p s) :
    UniformCauchySeqOn (fun n => g ∘ F n) p s := fun _u hu => hf _ (hg hu)


theorem UniformCauchySeqOn.prod_map {ι' α' β' : Type*} [UniformSpace β'] {F' : ι' → α' → β'}
    {p' : Filter ι'} {s' : Set α'} (h : UniformCauchySeqOn F p s)
    (h' : UniformCauchySeqOn F' p' s') :
    UniformCauchySeqOn (fun i : ι × ι' => Prod.map (F i.1) (F' i.2)) (p ×ˢ p') (s ×ˢ s') := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    ⊢ UniformCauchySeqOn (fun i => Prod.map (F i.1) (F' i.2)) (SProd.sprod p p') ( …
  -/
  intro u hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    hu : Membership.mem (uniformity (Prod β β')) u
    ⊢ Filter.Eventually (fun m => ∀ (x : Prod α α'), Membership.mem (SProd.sprod s …
  -/
  rw [uniformity_prod_eq_prod, mem_map, mem_prod_iff] at hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    hu : Exists fun t₁ => And (Membership.mem (uniformity β) t₁) (Exists fun t₂ => …
    ⊢ Filter.Eventually (fun m => ∀ (x : Prod α α'), Membership.mem (SProd.sprod s …
  -/
  obtain ⟨v, hv, w, hw, hvw⟩ := hu
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    v : Set (Prod β β)
    hv : Membership.mem (uniformity β) v
    w : Set (Prod β' β')
    hw : Membership.mem (uniformity β') w
    hvw : HasSubset.Subset (SProd.sprod v w) (Set.preimage (fun p => { fst := { fs …
    ⊢ Filter.Eventually (fun m => ∀ (x : Prod α α'), Membership.mem (SProd.sprod s …
  -/
  simp_rw [mem_prod, and_imp, Prod.forall, Prod.map_apply]
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    v : Set (Prod β β)
    hv : Membership.mem (uniformity β) v
    w : Set (Prod β' β')
    hw : Membership.mem (uniformity β') w
    hvw : HasSubset.Subset (SProd.sprod v w) (Set.preimage (fun p => { fst := { fs …
    ⊢ Filter.Eventually (fun m => ∀ (a : α) (b : α'), Membership.mem s a → Members …
  -/
  rw [← Set.image_subset_iff] at hvw
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    v : Set (Prod β β)
    hv : Membership.mem (uniformity β) v
    w : Set (Prod β' β')
    hw : Membership.mem (uniformity β') w
    hvw : HasSubset.Subset (Set.image (fun p => { fst := { fst := p.1.1, snd := p. …
    ⊢ Filter.Eventually (fun m => ∀ (a : α) (b : α'), Membership.mem s a → Members …
  -/
  apply (tendsto_swap4_prod.eventually ((h v hv).prod_mk (h' w hw))).mono
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    v : Set (Prod β β)
    hv : Membership.mem (uniformity β) v
    w : Set (Prod β' β')
    hw : Membership.mem (uniformity β') w
    hvw : HasSubset.Subset (Set.image (fun p => { fst := { fst := p.1.1, snd := p. …
    ⊢ ∀ (x : Prod (Prod ι ι') (Prod ι ι')), And (∀ (x_1 : α), Membership.mem s x_1 …
  -/
  intro x hx a b ha hb
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    s : Set α
    p : Filter ι
    ι' : Type u_1
    α' : Type u_2
    β' : Type u_3
    inst✝ : UniformSpace β'
    F' : ι' → α' → β'
    p' : Filter ι'
    s' : Set α'
    h : UniformCauchySeqOn F p s
    h' : UniformCauchySeqOn F' p' s'
    u : Set (Prod (Prod β β') (Prod β β'))
    v : Set (Prod β β)
    hv : Membership.mem (uniformity β) v
    w : Set (Prod β' β')
    hw : Membership.mem (uniformity β') w
    hvw : HasSubset.Subset (Set.image (fun p => { fst := { fst := p.1.1, snd := p. …
    x : Prod (Prod ι ι') (Prod ι ι')
    hx : And (∀ (x_1 : α), Membership.mem s x_1 → Membership.mem v { fst := F { fs …
    a : α
    b : α'
    ha : Membership.mem s a
    hb : Membership.mem s' b
    ⊢ Membership.mem u { fst := { fst := F x.1.1 a, snd := F' x.1.2 b }, snd := {  …
  -/
  exact hvw ⟨_, mk_mem_prod (hx.1 a ha) (hx.2 b hb), rfl⟩
  /-
    🎉 no goals
  -/


theorem UniformCauchySeqOn.prod {ι' β' : Type*} [UniformSpace β'] {F' : ι' → α → β'}
    {p' : Filter ι'} (h : UniformCauchySeqOn F p s) (h' : UniformCauchySeqOn F' p' s) :
    UniformCauchySeqOn (fun (i : ι × ι') a => (F i.fst a, F' i.snd a)) (p ×ˢ p') s :=
  (congr_arg _ s.inter_self).mp ((h.prod_map h').comp fun a => (a, a))


theorem UniformCauchySeqOn.prod' {β' : Type*} [UniformSpace β'] {F' : ι → α → β'}
    (h : UniformCauchySeqOn F p s) (h' : UniformCauchySeqOn F' p s) :
    UniformCauchySeqOn (fun (i : ι) a => (F i a, F' i a)) p s := fun u hu =>
  have hh : Tendsto (fun x : ι => (x, x)) p (p ×ˢ p) := tendsto_diag
  (hh.prod_map hh).eventually ((h.prod h') u hu)


/-- If a sequence of functions is uniformly Cauchy on a set, then the values at each point form
a Cauchy sequence. -/
theorem UniformCauchySeqOn.cauchy_map [hp : NeBot p] (hf : UniformCauchySeqOn F p s) (hx : x ∈ s) :
    Cauchy (map (fun i => F i x) p) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    x : α
    p : Filter ι
    hp : p.NeBot
    hf : UniformCauchySeqOn F p s
    hx : Membership.mem s x
    ⊢ Cauchy (Filter.map (fun i => F i x) p)
  -/
  simp only [cauchy_map_iff, hp, true_and]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    x : α
    p : Filter ι
    hp : p.NeBot
    hf : UniformCauchySeqOn F p s
    hx : Membership.mem s x
    ⊢ Filter.Tendsto (fun p => { fst := F p.1 x, snd := F p.2 x }) (SProd.sprod p  …
  -/
  intro u hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    x : α
    p : Filter ι
    hp : p.NeBot
    hf : UniformCauchySeqOn F p s
    hx : Membership.mem s x
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Membership.mem (Filter.map (fun p => { fst := F p.1 x, snd := F p.2 x }) (SP …
  -/
  rw [mem_map]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    s : Set α
    x : α
    p : Filter ι
    hp : p.NeBot
    hf : UniformCauchySeqOn F p s
    hx : Membership.mem s x
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Membership.mem (SProd.sprod p p) (Set.preimage (fun p => { fst := F p.1 x, s …
  -/
  filter_upwards [hf u hu] with p hp using hp x hx
  /-
    🎉 no goals
  -/


/-- If a sequence of functions is uniformly Cauchy on a set, then the values at each point form
a Cauchy sequence.  See `UniformCauchSeqOn.cauchy_map` for the non-`atTop` case. -/
theorem UniformCauchySeqOn.cauchySeq [Nonempty ι] [SemilatticeSup ι]
    (hf : UniformCauchySeqOn F atTop s) (hx : x ∈ s) :
    CauchySeq fun i ↦ F i x :=
  hf.cauchy_map (hp := atTop_neBot) hx


theorem tendstoUniformlyOn_of_seq_tendstoUniformlyOn {l : Filter ι} [l.IsCountablyGenerated]
    (h : ∀ u : ℕ → ι, Tendsto u atTop l → TendstoUniformlyOn (fun n => F (u n)) f atTop s) :
    TendstoUniformlyOn F f l s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    h : ∀ (u : Nat → ι), Filter.Tendsto u Filter.atTop l → TendstoUniformlyOn (fun …
    ⊢ TendstoUniformlyOn F f l s
  -/
  rw [tendstoUniformlyOn_iff_tendsto, tendsto_iff_seq_tendsto]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    h : ∀ (u : Nat → ι), Filter.Tendsto u Filter.atTop l → TendstoUniformlyOn (fun …
    ⊢ ∀ (x : Nat → Prod ι α), Filter.Tendsto x Filter.atTop (SProd.sprod l (Filter …
  -/
  intro u hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    h : ∀ (u : Nat → ι), Filter.Tendsto u Filter.atTop l → TendstoUniformlyOn (fun …
    u : Nat → Prod ι α
    hu : Filter.Tendsto u Filter.atTop (SProd.sprod l (Filter.principal s))
    ⊢ Filter.Tendsto (Function.comp (fun q => { fst := f q.2, snd := F q.1 q.2 })  …
  -/
  rw [tendsto_prod_iff'] at hu
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    h : ∀ (u : Nat → ι), Filter.Tendsto u Filter.atTop l → TendstoUniformlyOn (fun …
    u : Nat → Prod ι α
    hu : And (Filter.Tendsto (fun n => (u n).1) Filter.atTop l) (Filter.Tendsto (f …
    ⊢ Filter.Tendsto (Function.comp (fun q => { fst := f q.2, snd := F q.1 q.2 })  …
  -/
  specialize h (fun n => (u n).fst) hu.1
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    u : Nat → Prod ι α
    hu : And (Filter.Tendsto (fun n => (u n).1) Filter.atTop l) (Filter.Tendsto (f …
    h : TendstoUniformlyOn (fun n => F (u n).1) f Filter.atTop s
    ⊢ Filter.Tendsto (Function.comp (fun q => { fst := f q.2, snd := F q.1 q.2 })  …
  -/
  rw [tendstoUniformlyOn_iff_tendsto] at h
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    u : Nat → Prod ι α
    hu : And (Filter.Tendsto (fun n => (u n).1) Filter.atTop l) (Filter.Tendsto (f …
    h : Filter.Tendsto (fun q => { fst := f q.2, snd := F (u q.1).1 q.2 }) (SProd. …
    ⊢ Filter.Tendsto (Function.comp (fun q => { fst := f q.2, snd := F q.1 q.2 })  …
  -/
  exact h.comp (tendsto_id.prod_mk hu.2)
  /-
    🎉 no goals
  -/


theorem TendstoUniformlyOn.seq_tendstoUniformlyOn {l : Filter ι} (h : TendstoUniformlyOn F f l s)
    (u : ℕ → ι) (hu : Tendsto u atTop l) : TendstoUniformlyOn (fun n => F (u n)) f atTop s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    h : TendstoUniformlyOn F f l s
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    ⊢ TendstoUniformlyOn (fun n => F (u n)) f Filter.atTop s
  -/
  rw [tendstoUniformlyOn_iff_tendsto] at h ⊢
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    l : Filter ι
    h : Filter.Tendsto (fun q => { fst := f q.2, snd := F q.1 q.2 }) (SProd.sprod  …
    u : Nat → ι
    hu : Filter.Tendsto u Filter.atTop l
    ⊢ Filter.Tendsto (fun q => { fst := f q.2, snd := F (u q.1) q.2 }) (SProd.spro …
  -/
  exact h.comp ((hu.comp tendsto_fst).prod_mk tendsto_snd)
  /-
    🎉 no goals
  -/


theorem tendstoUniformlyOn_iff_seq_tendstoUniformlyOn {l : Filter ι} [l.IsCountablyGenerated] :
    TendstoUniformlyOn F f l s ↔
      ∀ u : ℕ → ι, Tendsto u atTop l → TendstoUniformlyOn (fun n => F (u n)) f atTop s :=
  ⟨TendstoUniformlyOn.seq_tendstoUniformlyOn, tendstoUniformlyOn_of_seq_tendstoUniformlyOn⟩


theorem tendstoUniformly_iff_seq_tendstoUniformly {l : Filter ι} [l.IsCountablyGenerated] :
    TendstoUniformly F f l ↔
      ∀ u : ℕ → ι, Tendsto u atTop l → TendstoUniformly (fun n => F (u n)) f atTop := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    ⊢ Iff (TendstoUniformly F f l) (∀ (u : Nat → ι), Filter.Tendsto u Filter.atTop …
  -/
  simp_rw [← tendstoUniformlyOn_univ]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    ⊢ Iff (TendstoUniformlyOn F f l Set.univ) (∀ (u : Nat → ι), Filter.Tendsto u F …
  -/
  exact tendstoUniformlyOn_iff_seq_tendstoUniformlyOn
  /-
    🎉 no goals
  -/


theorem TendstoUniformlyOnFilter.tendsto_of_eventually_tendsto
    (h1 : TendstoUniformlyOnFilter F f p p') (h2 : ∀ᶠ i in p, Tendsto (F i) p' (𝓝 (L i)))
    (h3 : Tendsto L p (𝓝 ℓ)) : Tendsto f p' (𝓝 ℓ) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    ⊢ Filter.Tendsto f p' (nhds ℓ)
  -/
  rw [tendsto_nhds_left]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    ⊢ Filter.Tendsto (fun x => { fst := f x, snd := ℓ }) p' (uniformity β)
  -/
  intro s hs
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ⊢ Membership.mem (Filter.map (fun x => { fst := f x, snd := ℓ }) p') s
  -/
  rw [mem_map, Set.preimage, ← eventually_iff]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ⊢ Filter.Eventually (fun x => Membership.mem s { fst := f x, snd := ℓ }) p'
  -/
  obtain ⟨t, ht, hts⟩ := comp3_mem_uniformity hs
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    hts : HasSubset.Subset (compRel t (compRel t t)) s
    ⊢ Filter.Eventually (fun x => Membership.mem s { fst := f x, snd := ℓ }) p'
  -/
  have p1 : ∀ᶠ i in p, (L i, ℓ) ∈ t := tendsto_nhds_left.mp h3 ht
  have p2 : ∀ᶠ i in p, ∀ᶠ x in p', (F i x, L i) ∈ t := by
    filter_upwards [h2] with i h2 using tendsto_nhds_left.mp h2 ht
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    hts : HasSubset.Subset (compRel t (compRel t t)) s
    p1 : Filter.Eventually (fun i => Membership.mem t { fst := L i, snd := ℓ }) p
    p2 : Filter.Eventually (fun i => Filter.Eventually (fun x => Membership.mem t  …
    ⊢ Filter.Eventually (fun x => Membership.mem s { fst := f x, snd := ℓ }) p'
  -/
  have p3 : ∀ᶠ i in p, ∀ᶠ x in p', (f x, F i x) ∈ t := (h1 t ht).curry
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    hts : HasSubset.Subset (compRel t (compRel t t)) s
    p1 : Filter.Eventually (fun i => Membership.mem t { fst := L i, snd := ℓ }) p
    p2 : Filter.Eventually (fun i => Filter.Eventually (fun x => Membership.mem t  …
    p3 : Filter.Eventually (fun i => Filter.Eventually (fun x => Membership.mem t  …
    ⊢ Filter.Eventually (fun x => Membership.mem s { fst := f x, snd := ℓ }) p'
  -/
  obtain ⟨i, p4, p5, p6⟩ := (p1.and (p2.and p3)).exists
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    p' : Filter α
    inst✝ : p.NeBot
    L : ι → β
    ℓ : β
    h1 : TendstoUniformlyOnFilter F f p p'
    h2 : Filter.Eventually (fun i => Filter.Tendsto (F i) p' (nhds (L i))) p
    h3 : Filter.Tendsto L p (nhds ℓ)
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    hts : HasSubset.Subset (compRel t (compRel t t)) s
    p1 : Filter.Eventually (fun i => Membership.mem t { fst := L i, snd := ℓ }) p
    p2 : Filter.Eventually (fun i => Filter.Eventually (fun x => Membership.mem t  …
    p3 : Filter.Eventually (fun i => Filter.Eventually (fun x => Membership.mem t  …
    i : ι
    p4 : Membership.mem t { fst := L i, snd := ℓ }
    p5 : Filter.Eventually (fun x => Membership.mem t { fst := F i x, snd := L i } …
    p6 : Filter.Eventually (fun x => Membership.mem t { fst := f x, snd := F i x } …
    ⊢ Filter.Eventually (fun x => Membership.mem s { fst := f x, snd := ℓ }) p'
  -/
  filter_upwards [p5, p6] with x p5 p6 using hts ⟨F i x, p6, L i, p5, p4⟩
  /-
    🎉 no goals
  -/


theorem TendstoUniformly.tendsto_of_eventually_tendsto
    (h1 : TendstoUniformly F f p) (h2 : ∀ᶠ i in p, Tendsto (F i) p' (𝓝 (L i)))
    (h3 : Tendsto L p (𝓝 ℓ)) : Tendsto f p' (𝓝 ℓ) :=
  (h1.tendstoUniformlyOnFilter.mono_right le_top).tendsto_of_eventually_tendsto h2 h3


/-- A sequence of functions `Fₙ` converges locally uniformly on a set `s` to a limiting function
`f` with respect to a filter `p` if, for any entourage of the diagonal `u`, for any `x ∈ s`, one
has `p`-eventually `(f y, Fₙ y) ∈ u` for all `y` in a neighborhood of `x` in `s`. -/
def TendstoLocallyUniformlyOn (F : ι → α → β) (f : α → β) (p : Filter ι) (s : Set α) :=
  ∀ u ∈ 𝓤 β, ∀ x ∈ s, ∃ t ∈ 𝓝[s] x, ∀ᶠ n in p, ∀ y ∈ t, (f y, F n y) ∈ u


/-- A sequence of functions `Fₙ` converges locally uniformly to a limiting function `f` with respect
to a filter `p` if, for any entourage of the diagonal `u`, for any `x`, one has `p`-eventually
`(f y, Fₙ y) ∈ u` for all `y` in a neighborhood of `x`. -/
def TendstoLocallyUniformly (F : ι → α → β) (f : α → β) (p : Filter ι) :=
  ∀ u ∈ 𝓤 β, ∀ x : α, ∃ t ∈ 𝓝 x, ∀ᶠ n in p, ∀ y ∈ t, (f y, F n y) ∈ u


theorem tendstoLocallyUniformlyOn_univ :
    TendstoLocallyUniformlyOn F f p univ ↔ TendstoLocallyUniformly F f p := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝ : TopologicalSpace α
    ⊢ Iff (TendstoLocallyUniformlyOn F f p Set.univ) (TendstoLocallyUniformly F f p)
  -/
  simp [TendstoLocallyUniformlyOn, TendstoLocallyUniformly, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem tendstoLocallyUniformlyOn_iff_forall_tendsto :
    TendstoLocallyUniformlyOn F f p s ↔
      ∀ x ∈ s, Tendsto (fun y : ι × α => (f y.2, F y.1 y.2)) (p ×ˢ 𝓝[s] x) (𝓤 β) :=
  forall₂_swap.trans <| forall₄_congr fun _ _ _ _ => by
    /-
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      x✝³ : α
      x✝² : Membership.mem s x✝³
      x✝¹ : Set (Prod β β)
      x✝ : Membership.mem (uniformity β) x✝¹
      ⊢ Iff (Exists fun t => And (Membership.mem (nhdsWithin x✝³ s) t) (Filter.Event …
    -/
    rw [mem_map, mem_prod_iff_right]; rfl
                                      /-
                                        🎉 no goals
                                      -/


nonrec theorem IsOpen.tendstoLocallyUniformlyOn_iff_forall_tendsto (hs : IsOpen s) :
    TendstoLocallyUniformlyOn F f p s ↔
      ∀ x ∈ s, Tendsto (fun y : ι × α => (f y.2, F y.1 y.2)) (p ×ˢ 𝓝 x) (𝓤 β) :=
  tendstoLocallyUniformlyOn_iff_forall_tendsto.trans <| forall₂_congr fun x hx => by
    /-
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      hs : IsOpen s
      x : α
      hx : Membership.mem s x
      ⊢ Iff (Filter.Tendsto (fun y => { fst := f y.2, snd := F y.1 y.2 }) (SProd.spr …
    -/
    rw [hs.nhdsWithin_eq hx]
    /-
      🎉 no goals
    -/


theorem tendstoLocallyUniformly_iff_forall_tendsto :
    TendstoLocallyUniformly F f p ↔
      ∀ x, Tendsto (fun y : ι × α => (f y.2, F y.1 y.2)) (p ×ˢ 𝓝 x) (𝓤 β) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝ : TopologicalSpace α
    ⊢ Iff (TendstoLocallyUniformly F f p) (∀ (x : α), Filter.Tendsto (fun y => { f …
  -/
  simp [← tendstoLocallyUniformlyOn_univ, isOpen_univ.tendstoLocallyUniformlyOn_iff_forall_tendsto]
  /-
    🎉 no goals
  -/


theorem tendstoLocallyUniformlyOn_iff_tendstoLocallyUniformly_comp_coe :
    TendstoLocallyUniformlyOn F f p s ↔
      TendstoLocallyUniformly (fun i (x : s) => F i x) (f ∘ (↑)) p := by
  simp only [tendstoLocallyUniformly_iff_forall_tendsto, Subtype.forall', tendsto_map'_iff,
                                                                                           /-
                                                                                             α : Type u
                                                                                             β : Type v
                                                                                             ι : Type x
                                                                                             inst✝¹ : UniformSpace β
                                                                                             F : ι → α → β
                                                                                             f : α → β
                                                                                             s : Set α
                                                                                             p : Filter ι
                                                                                             inst✝ : TopologicalSpace α
                                                                                             ⊢ Iff (∀ (x : Subtype fun a => Membership.mem s a), Filter.Tendsto (Function.c …
                                                                                           -/
    tendstoLocallyUniformlyOn_iff_forall_tendsto, ← map_nhds_subtype_val, prod_map_right]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


protected theorem TendstoUniformlyOn.tendstoLocallyUniformlyOn (h : TendstoUniformlyOn F f p s) :
    TendstoLocallyUniformlyOn F f p s := fun u hu _ _ =>
                              /-
                                α : Type u
                                β : Type v
                                ι : Type x
                                inst✝¹ : UniformSpace β
                                F : ι → α → β
                                f : α → β
                                s : Set α
                                p : Filter ι
                                inst✝ : TopologicalSpace α
                                h : TendstoUniformlyOn F f p s
                                u : Set (Prod β β)
                                hu : Membership.mem (uniformity β) u
                                x✝¹ : α
                                x✝ : Membership.mem s x✝¹
                                ⊢ Filter.Eventually (fun n => ∀ (y : α), Membership.mem s y → Membership.mem u …
                              -/
  ⟨s, self_mem_nhdsWithin, by simpa using h u hu⟩
                              /-
                                🎉 no goals
                              -/


protected theorem TendstoUniformly.tendstoLocallyUniformly (h : TendstoUniformly F f p) :
                                                                       /-
                                                                         α : Type u
                                                                         β : Type v
                                                                         ι : Type x
                                                                         inst✝¹ : UniformSpace β
                                                                         F : ι → α → β
                                                                         f : α → β
                                                                         p : Filter ι
                                                                         inst✝ : TopologicalSpace α
                                                                         h : TendstoUniformly F f p
                                                                         u : Set (Prod β β)
                                                                         hu : Membership.mem (uniformity β) u
                                                                         x✝ : α
                                                                         ⊢ Filter.Eventually (fun n => ∀ (y : α), Membership.mem Set.univ y → Membershi …
                                                                       -/
    TendstoLocallyUniformly F f p := fun u hu _ => ⟨univ, univ_mem, by simpa using h u hu⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem TendstoLocallyUniformlyOn.mono (h : TendstoLocallyUniformlyOn F f p s) (h' : s' ⊆ s) :
    TendstoLocallyUniformlyOn F f p s' := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s s' : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    h' : HasSubset.Subset s' s
    ⊢ TendstoLocallyUniformlyOn F f p s'
  -/
  intro u hu x hx
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s s' : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    h' : HasSubset.Subset s' s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s' x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s') t) (Filter.Eventually  …
  -/
  rcases h u hu x (h' hx) with ⟨t, ht, H⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s s' : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    h' : HasSubset.Subset s' s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s' x
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    H : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s') t) (Filter.Eventually  …
  -/
  exact ⟨t, nhdsWithin_mono x h' ht, H.mono fun n => id⟩
  /-
    🎉 no goals
  -/

-- Porting note: generalized from `Type` to `Sort`

theorem tendstoLocallyUniformlyOn_iUnion {ι' : Sort*} {S : ι' → Set α} (hS : ∀ i, IsOpen (S i))
    (h : ∀ i, TendstoLocallyUniformlyOn F f p (S i)) :
    TendstoLocallyUniformlyOn F f p (⋃ i, S i) :=
  (isOpen_iUnion hS).tendstoLocallyUniformlyOn_iff_forall_tendsto.2 fun _x hx =>
    let ⟨i, hi⟩ := mem_iUnion.1 hx
    (hS i).tendstoLocallyUniformlyOn_iff_forall_tendsto.1 (h i) _ hi


theorem tendstoLocallyUniformlyOn_biUnion {s : Set γ} {S : γ → Set α} (hS : ∀ i ∈ s, IsOpen (S i))
    (h : ∀ i ∈ s, TendstoLocallyUniformlyOn F f p (S i)) :
    TendstoLocallyUniformlyOn F f p (⋃ i ∈ s, S i) :=
  tendstoLocallyUniformlyOn_iUnion (fun i => isOpen_iUnion (hS i)) fun i =>
   tendstoLocallyUniformlyOn_iUnion (hS i) (h i)


theorem tendstoLocallyUniformlyOn_sUnion (S : Set (Set α)) (hS : ∀ s ∈ S, IsOpen s)
    (h : ∀ s ∈ S, TendstoLocallyUniformlyOn F f p s) : TendstoLocallyUniformlyOn F f p (⋃₀ S) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : Set α), Membership.mem S s → IsOpen s
    h : ∀ (s : Set α), Membership.mem S s → TendstoLocallyUniformlyOn F f p s
    ⊢ TendstoLocallyUniformlyOn F f p S.sUnion
  -/
  rw [sUnion_eq_biUnion]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝ : TopologicalSpace α
    S : Set (Set α)
    hS : ∀ (s : Set α), Membership.mem S s → IsOpen s
    h : ∀ (s : Set α), Membership.mem S s → TendstoLocallyUniformlyOn F f p s
    ⊢ TendstoLocallyUniformlyOn F f p (Set.iUnion fun i => Set.iUnion fun x => i)
  -/
  exact tendstoLocallyUniformlyOn_biUnion hS h
  /-
    🎉 no goals
  -/


theorem TendstoLocallyUniformlyOn.union {s₁ s₂ : Set α} (hs₁ : IsOpen s₁) (hs₂ : IsOpen s₂)
    (h₁ : TendstoLocallyUniformlyOn F f p s₁) (h₂ : TendstoLocallyUniformlyOn F f p s₂) :
    TendstoLocallyUniformlyOn F f p (s₁ ∪ s₂) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝ : TopologicalSpace α
    s₁ s₂ : Set α
    hs₁ : IsOpen s₁
    hs₂ : IsOpen s₂
    h₁ : TendstoLocallyUniformlyOn F f p s₁
    h₂ : TendstoLocallyUniformlyOn F f p s₂
    ⊢ TendstoLocallyUniformlyOn F f p (Union.union s₁ s₂)
  -/
  rw [← sUnion_pair]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝ : TopologicalSpace α
    s₁ s₂ : Set α
    hs₁ : IsOpen s₁
    hs₂ : IsOpen s₂
    h₁ : TendstoLocallyUniformlyOn F f p s₁
    h₂ : TendstoLocallyUniformlyOn F f p s₂
    ⊢ TendstoLocallyUniformlyOn F f p (Insert.insert s₁ (Singleton.singleton s₂)). …
  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  refine tendstoLocallyUniformlyOn_sUnion _ ?_ ?_ <;> simp [*]
                                                      /-
                                                        🎉 no goals
                                                      -/

-- Porting note: tendstoLocallyUniformlyOn_univ moved up


protected theorem TendstoLocallyUniformly.tendstoLocallyUniformlyOn
    (h : TendstoLocallyUniformly F f p) : TendstoLocallyUniformlyOn F f p s :=
  (tendstoLocallyUniformlyOn_univ.mpr h).mono (subset_univ _)


/-- On a compact space, locally uniform convergence is just uniform convergence. -/
theorem tendstoLocallyUniformly_iff_tendstoUniformly_of_compactSpace [CompactSpace α] :
    TendstoLocallyUniformly F f p ↔ TendstoUniformly F f p := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    ⊢ Iff (TendstoLocallyUniformly F f p) (TendstoUniformly F f p)
  -/
  refine ⟨fun h V hV => ?_, TendstoUniformly.tendstoLocallyUniformly⟩
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem V { fst := f x, snd := …
  -/
  choose U hU using h V hV
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    hU : ∀ (x : α), And (Membership.mem (nhds x) (U x)) (Filter.Eventually (fun n  …
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem V { fst := f x, snd := …
  -/
  obtain ⟨t, ht⟩ := isCompact_univ.elim_nhds_subcover' (fun k _ => U k) fun k _ => (hU k).1
  /-
    case intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    hU : ∀ (x : α), And (Membership.mem (nhds x) (U x)) (Filter.Eventually (fun n  …
    t : Finset ↑Set.univ
    ht : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem V { fst := f x, snd := …
  -/
  replace hU := fun x : t => (hU x).2
  /-
    case intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    t : Finset ↑Set.univ
    ht : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
    hU : ∀ (x : Subtype fun x => Membership.mem t x), Filter.Eventually (fun n =>  …
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem V { fst := f x, snd := …
  -/
  rw [← eventually_all] at hU
  /-
    case intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    t : Finset ↑Set.univ
    ht : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
    hU : Filter.Eventually (fun x => ∀ (i : Subtype fun x => Membership.mem t x) ( …
    ⊢ Filter.Eventually (fun n => ∀ (x : α), Membership.mem V { fst := f x, snd := …
  -/
  refine hU.mono fun i hi x => ?_
  /-
    case intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    t : Finset ↑Set.univ
    ht : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U ↑x)
    hU : Filter.Eventually (fun x => ∀ (i : Subtype fun x => Membership.mem t x) ( …
    i : ι
    hi : ∀ (i_1 : Subtype fun x => Membership.mem t x) (y : α), Membership.mem (U  …
    x : α
    ⊢ Membership.mem V { fst := f x, snd := F i x }
  -/
  specialize ht (mem_univ x)
  /-
    case intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    t : Finset ↑Set.univ
    hU : Filter.Eventually (fun x => ∀ (i : Subtype fun x => Membership.mem t x) ( …
    i : ι
    hi : ∀ (i_1 : Subtype fun x => Membership.mem t x) (y : α), Membership.mem (U  …
    x : α
    ht : Membership.mem (Set.iUnion fun x => Set.iUnion fun h => U ↑x) x
    ⊢ Membership.mem V { fst := f x, snd := F i x }
  -/
  simp only [exists_prop, mem_iUnion, SetCoe.exists, exists_and_right, Subtype.coe_mk] at ht
  /-
    case intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    t : Finset ↑Set.univ
    hU : Filter.Eventually (fun x => ∀ (i : Subtype fun x => Membership.mem t x) ( …
    i : ι
    hi : ∀ (i_1 : Subtype fun x => Membership.mem t x) (y : α), Membership.mem (U  …
    x : α
    ht : Exists fun x_1 => And (Exists fun x => Membership.mem t ⟨x_1, ⋯⟩) (Member …
    ⊢ Membership.mem V { fst := f x, snd := F i x }
  -/
  obtain ⟨y, ⟨hy₁, hy₂⟩, hy₃⟩ := ht
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : CompactSpace α
    h : TendstoLocallyUniformly F f p
    V : Set (Prod β β)
    hV : Membership.mem (uniformity β) V
    U : α → Set α
    t : Finset ↑Set.univ
    hU : Filter.Eventually (fun x => ∀ (i : Subtype fun x => Membership.mem t x) ( …
    i : ι
    hi : ∀ (i_1 : Subtype fun x => Membership.mem t x) (y : α), Membership.mem (U  …
    x y : α
    hy₃ : Membership.mem (U y) x
    hy₁ : Membership.mem Set.univ y
    hy₂ : Membership.mem t ⟨y, ⋯⟩
    ⊢ Membership.mem V { fst := f x, snd := F i x }
  -/
  exact hi ⟨⟨y, hy₁⟩, hy₂⟩ x hy₃
  /-
    🎉 no goals
  -/


/-- For a compact set `s`, locally uniform convergence on `s` is just uniform convergence on `s`. -/
theorem tendstoLocallyUniformlyOn_iff_tendstoUniformlyOn_of_compact (hs : IsCompact s) :
    TendstoLocallyUniformlyOn F f p s ↔ TendstoUniformlyOn F f p s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    hs : IsCompact s
    ⊢ Iff (TendstoLocallyUniformlyOn F f p s) (TendstoUniformlyOn F f p s)
  -/
  haveI : CompactSpace s := isCompact_iff_compactSpace.mp hs
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    hs : IsCompact s
    this : CompactSpace ↑s
    ⊢ Iff (TendstoLocallyUniformlyOn F f p s) (TendstoUniformlyOn F f p s)
  -/
  refine ⟨fun h => ?_, TendstoUniformlyOn.tendstoLocallyUniformlyOn⟩
  rwa [tendstoLocallyUniformlyOn_iff_tendstoLocallyUniformly_comp_coe,
    tendstoLocallyUniformly_iff_tendstoUniformly_of_compactSpace, ←
    tendstoUniformlyOn_iff_tendstoUniformly_comp_coe] at h


theorem TendstoLocallyUniformlyOn.comp [TopologicalSpace γ] {t : Set γ}
    (h : TendstoLocallyUniformlyOn F f p s) (g : γ → α) (hg : MapsTo g t s)
    (cg : ContinuousOn g t) : TendstoLocallyUniformlyOn (fun n => F n ∘ g) (f ∘ g) p t := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace γ
    t : Set γ
    h : TendstoLocallyUniformlyOn F f p s
    g : γ → α
    hg : Set.MapsTo g t s
    cg : ContinuousOn g t
    ⊢ TendstoLocallyUniformlyOn (fun n => Function.comp (F n) g) (Function.comp f  …
  -/
  intro u hu x hx
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace γ
    t : Set γ
    h : TendstoLocallyUniformlyOn F f p s
    g : γ → α
    hg : Set.MapsTo g t s
    cg : ContinuousOn g t
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : γ
    hx : Membership.mem t x
    ⊢ Exists fun t_1 => And (Membership.mem (nhdsWithin x t) t_1) (Filter.Eventual …
  -/
  rcases h u hu (g x) (hg hx) with ⟨a, ha, H⟩
  have : g ⁻¹' a ∈ 𝓝[t] x :=
    (cg x hx).preimage_mem_nhdsWithin' (nhdsWithin_mono (g x) hg.image_subset ha)
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace γ
    t : Set γ
    h : TendstoLocallyUniformlyOn F f p s
    g : γ → α
    hg : Set.MapsTo g t s
    cg : ContinuousOn g t
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : γ
    hx : Membership.mem t x
    a : Set α
    ha : Membership.mem (nhdsWithin (g x) s) a
    H : Filter.Eventually (fun n => ∀ (y : α), Membership.mem a y → Membership.mem …
    this : Membership.mem (nhdsWithin x t) (Set.preimage g a)
    ⊢ Exists fun t_1 => And (Membership.mem (nhdsWithin x t) t_1) (Filter.Eventual …
  -/
  exact ⟨g ⁻¹' a, this, H.mono fun n hn y hy => hn _ hy⟩
  /-
    🎉 no goals
  -/


theorem TendstoLocallyUniformly.comp [TopologicalSpace γ] (h : TendstoLocallyUniformly F f p)
    (g : γ → α) (cg : Continuous g) : TendstoLocallyUniformly (fun n => F n ∘ g) (f ∘ g) p := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace γ
    h : TendstoLocallyUniformly F f p
    g : γ → α
    cg : Continuous g
    ⊢ TendstoLocallyUniformly (fun n => Function.comp (F n) g) (Function.comp f g) p
  -/
  rw [← tendstoLocallyUniformlyOn_univ] at h ⊢
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace γ
    h : TendstoLocallyUniformlyOn F f p Set.univ
    g : γ → α
    cg : Continuous g
    ⊢ TendstoLocallyUniformlyOn (fun n => Function.comp (F n) g) (Function.comp f  …
  -/
  rw [continuous_iff_continuousOn_univ] at cg
  /-
    α : Type u
    β : Type v
    γ : Type w
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace γ
    h : TendstoLocallyUniformlyOn F f p Set.univ
    g : γ → α
    cg : ContinuousOn g Set.univ
    ⊢ TendstoLocallyUniformlyOn (fun n => Function.comp (F n) g) (Function.comp f  …
  -/
  exact h.comp _ (mapsTo_univ _ _) cg
  /-
    🎉 no goals
  -/


theorem tendstoLocallyUniformlyOn_TFAE [LocallyCompactSpace α] (G : ι → α → β) (g : α → β)
    (p : Filter ι) (hs : IsOpen s) :
    List.TFAE [
      TendstoLocallyUniformlyOn G g p s,
      ∀ K, K ⊆ s → IsCompact K → TendstoUniformlyOn G g p K,
      ∀ x ∈ s, ∃ v ∈ 𝓝[s] x, TendstoUniformlyOn G g p v] := by
  tfae_have 1 → 2
  | h, K, hK1, hK2 =>
    (tendstoLocallyUniformlyOn_iff_tendstoUniformlyOn_of_compact hK2).mp (h.mono hK1)
  tfae_have 2 → 3
  | h, x, hx => by
    obtain ⟨K, ⟨hK1, hK2⟩, hK3⟩ := (compact_basis_nhds x).mem_iff.mp (hs.mem_nhds hx)
    exact ⟨K, nhdsWithin_le_nhds hK1, h K hK3 hK2⟩
  tfae_have 3 → 1
  | h, u, hu, x, hx => by
    obtain ⟨v, hv1, hv2⟩ := h x hx
    exact ⟨v, hv1, hv2 u hu⟩
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    s : Set α
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyCompactSpace α
    G : ι → α → β
    g : α → β
    p : Filter ι
    hs : IsOpen s
    tfae_1_to_2 : TendstoLocallyUniformlyOn G g p s → ∀ (K : Set α), HasSubset.Sub …
    tfae_2_to_3 : (∀ (K : Set α), HasSubset.Subset K s → IsCompact K → TendstoUnif …
    tfae_3_to_1 : (∀ (x : α), Membership.mem s x → Exists fun v => And (Membership …
    ⊢ (List.cons (TendstoLocallyUniformlyOn G g p s) (List.cons (∀ (K : Set α), Ha …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem tendstoLocallyUniformlyOn_iff_forall_isCompact [LocallyCompactSpace α] (hs : IsOpen s) :
    TendstoLocallyUniformlyOn F f p s ↔ ∀ K, K ⊆ s → IsCompact K → TendstoUniformlyOn F f p K :=
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    inst✝ : LocallyCompactSpace α
    hs : IsOpen s
    ⊢ Eq ((List.cons (TendstoLocallyUniformlyOn F f p s) (List.cons (∀ (K : Set α) …
  -/
  /-
    🎉 no goals
  -/
  (tendstoLocallyUniformlyOn_TFAE F f p hs).out 0 1
  /-
    🎉 no goals
  -/


lemma tendstoLocallyUniformly_iff_forall_isCompact [LocallyCompactSpace α]  :
    TendstoLocallyUniformly F f p ↔ ∀ K : Set α, IsCompact K → TendstoUniformlyOn F f p K := by
  simp only [← tendstoLocallyUniformlyOn_univ,
    tendstoLocallyUniformlyOn_iff_forall_isCompact isOpen_univ, Set.subset_univ, forall_true_left]


theorem tendstoLocallyUniformlyOn_iff_filter :
    TendstoLocallyUniformlyOn F f p s ↔ ∀ x ∈ s, TendstoUniformlyOnFilter F f p (𝓝[s] x) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    ⊢ Iff (TendstoLocallyUniformlyOn F f p s) (∀ (x : α), Membership.mem s x → Ten …
  -/
  simp only [TendstoUniformlyOnFilter, eventually_prod_iff]
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    ⊢ Iff (TendstoLocallyUniformlyOn F f p s) (∀ (x : α), Membership.mem s x → ∀ ( …
  -/
  constructor
    /-
      case mp
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      ⊢ TendstoLocallyUniformlyOn F f p s → ∀ (x : α), Membership.mem s x → ∀ (u : S …
    -/
  · rintro h x hx u hu
    /-
      case mp
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      h : TendstoLocallyUniformlyOn F f p s
      x : α
      hx : Membership.mem s x
      u : Set (Prod β β)
      hu : Membership.mem (uniformity β) u
      ⊢ Exists fun pa => And (Filter.Eventually (fun x => pa x) p) (Exists fun pb => …
    -/
    obtain ⟨s, hs1, hs2⟩ := h u hu x hx
    /-
      case mp.intro.intro
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s✝ : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      h : TendstoLocallyUniformlyOn F f p s✝
      x : α
      hx : Membership.mem s✝ x
      u : Set (Prod β β)
      hu : Membership.mem (uniformity β) u
      s : Set α
      hs1 : Membership.mem (nhdsWithin x s✝) s
      hs2 : Filter.Eventually (fun n => ∀ (y : α), Membership.mem s y → Membership.m …
      ⊢ Exists fun pa => And (Filter.Eventually (fun x => pa x) p) (Exists fun pb => …
    -/
    exact ⟨_, hs2, _, eventually_of_mem hs1 fun x => id, fun hi y hy => hi y hy⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      ⊢ (∀ (x : α), Membership.mem s x → ∀ (u : Set (Prod β β)), Membership.mem (uni …
    -/
  · rintro h u hu x hx
    /-
      case mpr
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      h : ∀ (x : α), Membership.mem s x → ∀ (u : Set (Prod β β)), Membership.mem (un …
      u : Set (Prod β β)
      hu : Membership.mem (uniformity β) u
      x : α
      hx : Membership.mem s x
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
    -/
    obtain ⟨pa, hpa, pb, hpb, h⟩ := h x hx u hu
    /-
      case mpr.intro.intro.intro.intro
      α : Type u
      β : Type v
      ι : Type x
      inst✝¹ : UniformSpace β
      F : ι → α → β
      f : α → β
      s : Set α
      p : Filter ι
      inst✝ : TopologicalSpace α
      h✝ : ∀ (x : α), Membership.mem s x → ∀ (u : Set (Prod β β)), Membership.mem (u …
      u : Set (Prod β β)
      hu : Membership.mem (uniformity β) u
      x : α
      hx : Membership.mem s x
      pa : ι → Prop
      hpa : Filter.Eventually (fun x => pa x) p
      pb : α → Prop
      hpb : Filter.Eventually (fun y => pb y) (nhdsWithin x s)
      h : ∀ {x : ι}, pa x → ∀ {y : α}, pb y → Membership.mem u { fst := f y, snd :=  …
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
    -/
    exact ⟨pb, hpb, eventually_of_mem hpa fun i hi y hy => h hi hy⟩
    /-
      🎉 no goals
    -/


theorem tendstoLocallyUniformly_iff_filter :
    TendstoLocallyUniformly F f p ↔ ∀ x, TendstoUniformlyOnFilter F f p (𝓝 x) := by
  simpa [← tendstoLocallyUniformlyOn_univ, ← nhdsWithin_univ] using
    @tendstoLocallyUniformlyOn_iff_filter _ _ _ _ F f univ p _


theorem TendstoLocallyUniformlyOn.tendsto_at (hf : TendstoLocallyUniformlyOn F f p s) {a : α}
    (ha : a ∈ s) : Tendsto (fun i => F i a) p (𝓝 (f a)) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    hf : TendstoLocallyUniformlyOn F f p s
    a : α
    ha : Membership.mem s a
    ⊢ Filter.Tendsto (fun i => F i a) p (nhds (f a))
  -/
  refine ((tendstoLocallyUniformlyOn_iff_filter.mp hf) a ha).tendsto_at ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    hf : TendstoLocallyUniformlyOn F f p s
    a : α
    ha : Membership.mem s a
    ⊢ LE.le (Filter.principal (Singleton.singleton a)) (nhdsWithin a s)
  -/
  simpa only [Filter.principal_singleton] using pure_le_nhdsWithin ha
  /-
    🎉 no goals
  -/


theorem TendstoLocallyUniformlyOn.unique [p.NeBot] [T2Space β] {g : α → β}
    (hf : TendstoLocallyUniformlyOn F f p s) (hg : TendstoLocallyUniformlyOn F g p s) :
    s.EqOn f g := fun _a ha => tendsto_nhds_unique (hf.tendsto_at ha) (hg.tendsto_at ha)


theorem TendstoLocallyUniformlyOn.congr {G : ι → α → β} (hf : TendstoLocallyUniformlyOn F f p s)
    (hg : ∀ n, s.EqOn (F n) (G n)) : TendstoLocallyUniformlyOn G f p s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    G : ι → α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : ∀ (n : ι), Set.EqOn (F n) (G n) s
    ⊢ TendstoLocallyUniformlyOn G f p s
  -/
  rintro u hu x hx
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    G : ι → α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : ∀ (n : ι), Set.EqOn (F n) (G n) s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  obtain ⟨t, ht, h⟩ := hf u hu x hx
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    G : ι → α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : ∀ (n : ι), Set.EqOn (F n) (G n) s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s x
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    h : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  refine ⟨s ∩ t, inter_mem self_mem_nhdsWithin ht, ?_⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    G : ι → α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : ∀ (n : ι), Set.EqOn (F n) (G n) s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s x
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    h : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    ⊢ Filter.Eventually (fun n => ∀ (y : α), Membership.mem (Inter.inter s t) y →  …
  -/
  filter_upwards [h] with i hi y hy using hg i hy.1 ▸ hi y hy.2
  /-
    🎉 no goals
  -/


theorem TendstoLocallyUniformlyOn.congr_right {g : α → β} (hf : TendstoLocallyUniformlyOn F f p s)
    (hg : s.EqOn f g) : TendstoLocallyUniformlyOn F g p s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    g : α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : Set.EqOn f g s
    ⊢ TendstoLocallyUniformlyOn F g p s
  -/
  rintro u hu x hx
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    g : α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : Set.EqOn f g s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  obtain ⟨t, ht, h⟩ := hf u hu x hx
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    g : α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : Set.EqOn f g s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s x
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    h : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Filter.Eventually ( …
  -/
  refine ⟨s ∩ t, inter_mem self_mem_nhdsWithin ht, ?_⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝ : TopologicalSpace α
    g : α → β
    hf : TendstoLocallyUniformlyOn F f p s
    hg : Set.EqOn f g s
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    x : α
    hx : Membership.mem s x
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    h : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    ⊢ Filter.Eventually (fun n => ∀ (y : α), Membership.mem (Inter.inter s t) y →  …
  -/
  filter_upwards [h] with i hi y hy using hg hy.1 ▸ hi y hy.2
  /-
    🎉 no goals
  -/


/-- A function which can be locally uniformly approximated by functions which are continuous
within a set at a point is continuous within this set at this point. -/
theorem continuousWithinAt_of_locally_uniform_approx_of_continuousWithinAt (hx : x ∈ s)
    (L : ∀ u ∈ 𝓤 β, ∃ t ∈ 𝓝[s] x, ∃ F : α → β, ContinuousWithinAt F s x ∧ ∀ y ∈ t, (f y, F y) ∈ u) :
    ContinuousWithinAt f s x := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    s : Set α
    x : α
    inst✝ : TopologicalSpace α
    hx : Membership.mem s x
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    ⊢ ContinuousWithinAt f s x
  -/
  refine Uniform.continuousWithinAt_iff'_left.2 fun u₀ hu₀ => ?_
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    s : Set α
    x : α
    inst✝ : TopologicalSpace α
    hx : Membership.mem s x
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x_1, snd := f x }) (nhdsWi …
  -/
  obtain ⟨u₁, h₁, u₁₀⟩ : ∃ u ∈ 𝓤 β, u ○ u ⊆ u₀ := comp_mem_uniformity_sets hu₀
  obtain ⟨u₂, h₂, hsymm, u₂₁⟩ : ∃ u ∈ 𝓤 β, (∀ {a b}, (a, b) ∈ u → (b, a) ∈ u) ∧ u ○ u ⊆ u₁ :=
    comp_symm_of_uniformity h₁
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    s : Set α
    x : α
    inst✝ : TopologicalSpace α
    hx : Membership.mem s x
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    u₂ : Set (Prod β β)
    h₂ : Membership.mem (uniformity β) u₂
    hsymm : ∀ {a b : β}, Membership.mem u₂ { fst := a, snd := b } → Membership.mem …
    u₂₁ : HasSubset.Subset (compRel u₂ u₂) u₁
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x_1, snd := f x }) (nhdsWi …
  -/
  rcases L u₂ h₂ with ⟨t, tx, F, hFc, hF⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    s : Set α
    x : α
    inst✝ : TopologicalSpace α
    hx : Membership.mem s x
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    u₂ : Set (Prod β β)
    h₂ : Membership.mem (uniformity β) u₂
    hsymm : ∀ {a b : β}, Membership.mem u₂ { fst := a, snd := b } → Membership.mem …
    u₂₁ : HasSubset.Subset (compRel u₂ u₂) u₁
    t : Set α
    tx : Membership.mem (nhdsWithin x s) t
    F : α → β
    hFc : ContinuousWithinAt F s x
    hF : ∀ (y : α), Membership.mem t y → Membership.mem u₂ { fst := f y, snd := F  …
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x_1, snd := f x }) (nhdsWi …
  -/
  have A : ∀ᶠ y in 𝓝[s] x, (f y, F y) ∈ u₂ := Eventually.mono tx hF
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    s : Set α
    x : α
    inst✝ : TopologicalSpace α
    hx : Membership.mem s x
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    u₂ : Set (Prod β β)
    h₂ : Membership.mem (uniformity β) u₂
    hsymm : ∀ {a b : β}, Membership.mem u₂ { fst := a, snd := b } → Membership.mem …
    u₂₁ : HasSubset.Subset (compRel u₂ u₂) u₁
    t : Set α
    tx : Membership.mem (nhdsWithin x s) t
    F : α → β
    hFc : ContinuousWithinAt F s x
    hF : ∀ (y : α), Membership.mem t y → Membership.mem u₂ { fst := f y, snd := F  …
    A : Filter.Eventually (fun y => Membership.mem u₂ { fst := f y, snd := F y })  …
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x_1, snd := f x }) (nhdsWi …
  -/
  have B : ∀ᶠ y in 𝓝[s] x, (F y, F x) ∈ u₂ := Uniform.continuousWithinAt_iff'_left.1 hFc h₂
  have C : ∀ᶠ y in 𝓝[s] x, (f y, F x) ∈ u₁ :=
    (A.and B).mono fun y hy => u₂₁ (prod_mk_mem_compRel hy.1 hy.2)
  have : (F x, f x) ∈ u₁ :=
    u₂₁ (prod_mk_mem_compRel (refl_mem_uniformity h₂) (hsymm (A.self_of_nhdsWithin hx)))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    s : Set α
    x : α
    inst✝ : TopologicalSpace α
    hx : Membership.mem s x
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    u₂ : Set (Prod β β)
    h₂ : Membership.mem (uniformity β) u₂
    hsymm : ∀ {a b : β}, Membership.mem u₂ { fst := a, snd := b } → Membership.mem …
    u₂₁ : HasSubset.Subset (compRel u₂ u₂) u₁
    t : Set α
    tx : Membership.mem (nhdsWithin x s) t
    F : α → β
    hFc : ContinuousWithinAt F s x
    hF : ∀ (y : α), Membership.mem t y → Membership.mem u₂ { fst := f y, snd := F  …
    A : Filter.Eventually (fun y => Membership.mem u₂ { fst := f y, snd := F y })  …
    B : Filter.Eventually (fun y => Membership.mem u₂ { fst := F y, snd := F x })  …
    C : Filter.Eventually (fun y => Membership.mem u₁ { fst := f y, snd := F x })  …
    this : Membership.mem u₁ { fst := F x, snd := f x }
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x_1, snd := f x }) (nhdsWi …
  -/
  exact C.mono fun y hy => u₁₀ (prod_mk_mem_compRel hy this)
  /-
    🎉 no goals
  -/


/-- A function which can be locally uniformly approximated by functions which are continuous at
a point is continuous at this point. -/
theorem continuousAt_of_locally_uniform_approx_of_continuousAt
    (L : ∀ u ∈ 𝓤 β, ∃ t ∈ 𝓝 x, ∃ F, ContinuousAt F x ∧ ∀ y ∈ t, (f y, F y) ∈ u) :
    ContinuousAt f x := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    x : α
    inst✝ : TopologicalSpace α
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    ⊢ ContinuousAt f x
  -/
  rw [← continuousWithinAt_univ]
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    x : α
    inst✝ : TopologicalSpace α
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    ⊢ ContinuousWithinAt f Set.univ x
  -/
  apply continuousWithinAt_of_locally_uniform_approx_of_continuousWithinAt (mem_univ _) _
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace β
    f : α → β
    x : α
    inst✝ : TopologicalSpace α
    L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t =>  …
    ⊢ ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t => An …
  -/
  simpa only [exists_prop, nhdsWithin_univ, continuousWithinAt_univ] using L
  /-
    🎉 no goals
  -/


/-- A function which can be locally uniformly approximated by functions which are continuous
on a set is continuous on this set. -/
theorem continuousOn_of_locally_uniform_approx_of_continuousWithinAt
    (L : ∀ x ∈ s, ∀ u ∈ 𝓤 β, ∃ t ∈ 𝓝[s] x, ∃ F,
      ContinuousWithinAt F s x ∧ ∀ y ∈ t, (f y, F y) ∈ u) :
    ContinuousOn f s := fun x hx =>
  continuousWithinAt_of_locally_uniform_approx_of_continuousWithinAt hx (L x hx)


/-- A function which can be uniformly approximated by functions which are continuous on a set
is continuous on this set. -/
theorem continuousOn_of_uniform_approx_of_continuousOn
    (L : ∀ u ∈ 𝓤 β, ∃ F, ContinuousOn F s ∧ ∀ y ∈ s, (f y, F y) ∈ u) : ContinuousOn f s :=
  continuousOn_of_locally_uniform_approx_of_continuousWithinAt fun _x hx u hu =>
    ⟨s, self_mem_nhdsWithin, (L u hu).imp fun _F hF => ⟨hF.1.continuousWithinAt hx, hF.2⟩⟩


/-- A function which can be locally uniformly approximated by continuous functions is continuous. -/
theorem continuous_of_locally_uniform_approx_of_continuousAt
    (L : ∀ x : α, ∀ u ∈ 𝓤 β, ∃ t ∈ 𝓝 x, ∃ F, ContinuousAt F x ∧ ∀ y ∈ t, (f y, F y) ∈ u) :
    Continuous f :=
  continuous_iff_continuousAt.2 fun x =>
    continuousAt_of_locally_uniform_approx_of_continuousAt (L x)


/-- A function which can be uniformly approximated by continuous functions is continuous. -/
theorem continuous_of_uniform_approx_of_continuous
    (L : ∀ u ∈ 𝓤 β, ∃ F, Continuous F ∧ ∀ y, (f y, F y) ∈ u) : Continuous f :=
  continuous_iff_continuousOn_univ.mpr <|
    continuousOn_of_uniform_approx_of_continuousOn <| by
      /-
        α : Type u
        β : Type v
        inst✝¹ : UniformSpace β
        f : α → β
        inst✝ : TopologicalSpace α
        L : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun F =>  …
        ⊢ ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun F => An …
      -/
      simpa [continuous_iff_continuousOn_univ] using L
      /-
        🎉 no goals
      -/


/-- A locally uniform limit on a set of functions which are continuous on this set is itself
continuous on this set. -/
protected theorem TendstoLocallyUniformlyOn.continuousOn (h : TendstoLocallyUniformlyOn F f p s)
    (hc : ∀ᶠ n in p, ContinuousOn (F n) s) [NeBot p] : ContinuousOn f s := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    hc : Filter.Eventually (fun n => ContinuousOn (F n) s) p
    inst✝ : p.NeBot
    ⊢ ContinuousOn f s
  -/
  refine continuousOn_of_locally_uniform_approx_of_continuousWithinAt fun x hx u hu => ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    hc : Filter.Eventually (fun n => ContinuousOn (F n) s) p
    inst✝ : p.NeBot
    x : α
    hx : Membership.mem s x
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Exists fun F => And …
  -/
  rcases h u hu x hx with ⟨t, ht, H⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    hc : Filter.Eventually (fun n => ContinuousOn (F n) s) p
    inst✝ : p.NeBot
    x : α
    hx : Membership.mem s x
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    H : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Exists fun F => And …
  -/
  rcases (hc.and H).exists with ⟨n, hFc, hF⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝² : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    p : Filter ι
    inst✝¹ : TopologicalSpace α
    h : TendstoLocallyUniformlyOn F f p s
    hc : Filter.Eventually (fun n => ContinuousOn (F n) s) p
    inst✝ : p.NeBot
    x : α
    hx : Membership.mem s x
    u : Set (Prod β β)
    hu : Membership.mem (uniformity β) u
    t : Set α
    ht : Membership.mem (nhdsWithin x s) t
    H : Filter.Eventually (fun n => ∀ (y : α), Membership.mem t y → Membership.mem …
    n : ι
    hFc : ContinuousOn (F n) s
    hF : ∀ (y : α), Membership.mem t y → Membership.mem u { fst := f y, snd := F n …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (Exists fun F => And …
  -/
  exact ⟨t, ht, ⟨F n, hFc.continuousWithinAt hx, hF⟩⟩
  /-
    🎉 no goals
  -/


/-- A uniform limit on a set of functions which are continuous on this set is itself continuous
on this set. -/
protected theorem TendstoUniformlyOn.continuousOn (h : TendstoUniformlyOn F f p s)
    (hc : ∀ᶠ n in p, ContinuousOn (F n) s) [NeBot p] : ContinuousOn f s :=
  h.tendstoLocallyUniformlyOn.continuousOn hc


/-- A locally uniform limit of continuous functions is continuous. -/
protected theorem TendstoLocallyUniformly.continuous (h : TendstoLocallyUniformly F f p)
    (hc : ∀ᶠ n in p, Continuous (F n)) [NeBot p] : Continuous f :=
  continuous_iff_continuousOn_univ.mpr <|
    h.tendstoLocallyUniformlyOn.continuousOn <| hc.mono fun _n hn => hn.continuousOn


/-- A uniform limit of continuous functions is continuous. -/
protected theorem TendstoUniformly.continuous (h : TendstoUniformly F f p)
    (hc : ∀ᶠ n in p, Continuous (F n)) [NeBot p] : Continuous f :=
  h.tendstoLocallyUniformly.continuous hc


/-- If `Fₙ` converges locally uniformly on a neighborhood of `x` within a set `s` to a function `f`
which is continuous at `x` within `s`, and `gₙ` tends to `x` within `s`, then `Fₙ (gₙ)` tends
to `f x`. -/
theorem tendsto_comp_of_locally_uniform_limit_within (h : ContinuousWithinAt f s x)
    (hg : Tendsto g p (𝓝[s] x))
    (hunif : ∀ u ∈ 𝓤 β, ∃ t ∈ 𝓝[s] x, ∀ᶠ n in p, ∀ y ∈ t, (f y, F n y) ∈ u) :
    Tendsto (fun n => F n (g n)) p (𝓝 (f x)) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f s x
    hg : Filter.Tendsto g p (nhdsWithin x s)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    ⊢ Filter.Tendsto (fun n => F n (g n)) p (nhds (f x))
  -/
  refine Uniform.tendsto_nhds_right.2 fun u₀ hu₀ => ?_
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f s x
    hg : Filter.Tendsto g p (nhdsWithin x s)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x, snd := F x_1 (g x_1) }) …
  -/
  obtain ⟨u₁, h₁, u₁₀⟩ : ∃ u ∈ 𝓤 β, u ○ u ⊆ u₀ := comp_mem_uniformity_sets hu₀
  /-
    case intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s : Set α
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f s x
    hg : Filter.Tendsto g p (nhdsWithin x s)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x, snd := F x_1 (g x_1) }) …
  -/
  rcases hunif u₁ h₁ with ⟨s, sx, hs⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s✝ : Set α
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f s✝ x
    hg : Filter.Tendsto g p (nhdsWithin x s✝)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    s : Set α
    sx : Membership.mem (nhdsWithin x s✝) s
    hs : Filter.Eventually (fun n => ∀ (y : α), Membership.mem s y → Membership.me …
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x, snd := F x_1 (g x_1) }) …
  -/
  have A : ∀ᶠ n in p, g n ∈ s := hg sx
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s✝ : Set α
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f s✝ x
    hg : Filter.Tendsto g p (nhdsWithin x s✝)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    s : Set α
    sx : Membership.mem (nhdsWithin x s✝) s
    hs : Filter.Eventually (fun n => ∀ (y : α), Membership.mem s y → Membership.me …
    A : Filter.Eventually (fun n => Membership.mem s (g n)) p
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x, snd := F x_1 (g x_1) }) …
  -/
  have B : ∀ᶠ n in p, (f x, f (g n)) ∈ u₁ := hg (Uniform.continuousWithinAt_iff'_right.1 h h₁)
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    s✝ : Set α
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f s✝ x
    hg : Filter.Tendsto g p (nhdsWithin x s✝)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    u₀ : Set (Prod β β)
    hu₀ : Membership.mem (uniformity β) u₀
    u₁ : Set (Prod β β)
    h₁ : Membership.mem (uniformity β) u₁
    u₁₀ : HasSubset.Subset (compRel u₁ u₁) u₀
    s : Set α
    sx : Membership.mem (nhdsWithin x s✝) s
    hs : Filter.Eventually (fun n => ∀ (y : α), Membership.mem s y → Membership.me …
    A : Filter.Eventually (fun n => Membership.mem s (g n)) p
    B : Filter.Eventually (fun n => Membership.mem u₁ { fst := f x, snd := f (g n) …
    ⊢ Membership.mem (Filter.map (fun x_1 => { fst := f x, snd := F x_1 (g x_1) }) …
  -/
  exact B.mp <| A.mp <| hs.mono fun y H1 H2 H3 => u₁₀ (prod_mk_mem_compRel H3 (H1 _ H2))
  /-
    🎉 no goals
  -/


/-- If `Fₙ` converges locally uniformly on a neighborhood of `x` to a function `f` which is
continuous at `x`, and `gₙ` tends to `x`, then `Fₙ (gₙ)` tends to `f x`. -/
theorem tendsto_comp_of_locally_uniform_limit (h : ContinuousAt f x) (hg : Tendsto g p (𝓝 x))
    (hunif : ∀ u ∈ 𝓤 β, ∃ t ∈ 𝓝 x, ∀ᶠ n in p, ∀ y ∈ t, (f y, F n y) ∈ u) :
    Tendsto (fun n => F n (g n)) p (𝓝 (f x)) := by
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousAt f x
    hg : Filter.Tendsto g p (nhds x)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    ⊢ Filter.Tendsto (fun n => F n (g n)) p (nhds (f x))
  -/
  rw [← continuousWithinAt_univ] at h
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f Set.univ x
    hg : Filter.Tendsto g p (nhds x)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    ⊢ Filter.Tendsto (fun n => F n (g n)) p (nhds (f x))
  -/
  rw [← nhdsWithin_univ] at hunif hg
  /-
    α : Type u
    β : Type v
    ι : Type x
    inst✝¹ : UniformSpace β
    F : ι → α → β
    f : α → β
    x : α
    p : Filter ι
    g : ι → α
    inst✝ : TopologicalSpace α
    h : ContinuousWithinAt f Set.univ x
    hg : Filter.Tendsto g p (nhdsWithin x Set.univ)
    hunif : ∀ (u : Set (Prod β β)), Membership.mem (uniformity β) u → Exists fun t …
    ⊢ Filter.Tendsto (fun n => F n (g n)) p (nhds (f x))
  -/
  exact tendsto_comp_of_locally_uniform_limit_within h hg hunif
  /-
    🎉 no goals
  -/


/-- If `Fₙ` tends locally uniformly to `f` on a set `s`, and `gₙ` tends to `x` within `s`, then
`Fₙ gₙ` tends to `f x` if `f` is continuous at `x` within `s` and `x ∈ s`. -/
theorem TendstoLocallyUniformlyOn.tendsto_comp (h : TendstoLocallyUniformlyOn F f p s)
    (hf : ContinuousWithinAt f s x) (hx : x ∈ s) (hg : Tendsto g p (𝓝[s] x)) :
    Tendsto (fun n => F n (g n)) p (𝓝 (f x)) :=
  tendsto_comp_of_locally_uniform_limit_within hf hg fun u hu => h u hu x hx


/-- If `Fₙ` tends uniformly to `f` on a set `s`, and `gₙ` tends to `x` within `s`, then `Fₙ gₙ`
tends to `f x` if `f` is continuous at `x` within `s`. -/
theorem TendstoUniformlyOn.tendsto_comp (h : TendstoUniformlyOn F f p s)
    (hf : ContinuousWithinAt f s x) (hg : Tendsto g p (𝓝[s] x)) :
    Tendsto (fun n => F n (g n)) p (𝓝 (f x)) :=
  tendsto_comp_of_locally_uniform_limit_within hf hg fun u hu => ⟨s, self_mem_nhdsWithin, h u hu⟩


/-- If `Fₙ` tends locally uniformly to `f`, and `gₙ` tends to `x`, then `Fₙ gₙ` tends to `f x`. -/
theorem TendstoLocallyUniformly.tendsto_comp (h : TendstoLocallyUniformly F f p)
    (hf : ContinuousAt f x) (hg : Tendsto g p (𝓝 x)) : Tendsto (fun n => F n (g n)) p (𝓝 (f x)) :=
  tendsto_comp_of_locally_uniform_limit hf hg fun u hu => h u hu x


/-- If `Fₙ` tends uniformly to `f`, and `gₙ` tends to `x`, then `Fₙ gₙ` tends to `f x`. -/
theorem TendstoUniformly.tendsto_comp (h : TendstoUniformly F f p) (hf : ContinuousAt f x)
    (hg : Tendsto g p (𝓝 x)) : Tendsto (fun n => F n (g n)) p (𝓝 (f x)) :=
  h.tendstoLocallyUniformly.tendsto_comp hf hg

