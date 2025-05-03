@[to_additive]
theorem mulIndicator_eventuallyEq (hf : f =ᶠ[l ⊓ 𝓟 s] g) (hs : s =ᶠ[l] t) :
    mulIndicator s f =ᶠ[l] mulIndicator t g :=
  (eventually_inf_principal.1 hf).mp <| hs.mem_iff.mono fun x hst hfg =>
    by_cases
                             /-
                               α : Type u_1
                               M : Type u_3
                               inst✝ : One M
                               s t : Set α
                               f g : α → M
                               l : Filter α
                               hf : (Min.min l (Filter.principal s)).EventuallyEq f g
                               hs : l.EventuallyEq s t
                               x : α
                               hst : Iff (Membership.mem s x) (Membership.mem t x)
                               hfg : Membership.mem s x → Eq (f x) (g x)
                               hxs : Membership.mem s x
                               ⊢ Eq (s.mulIndicator f x) (t.mulIndicator g x)
                             -/
      (fun hxs : x ∈ s => by simp only [*, hst.1 hxs, mulIndicator_of_mem])
                             /-
                               🎉 no goals
                             -/
                     /-
                       α : Type u_1
                       M : Type u_3
                       inst✝ : One M
                       s t : Set α
                       f g : α → M
                       l : Filter α
                       hf : (Min.min l (Filter.principal s)).EventuallyEq f g
                       hs : l.EventuallyEq s t
                       x : α
                       hst : Iff (Membership.mem s x) (Membership.mem t x)
                       hfg : Membership.mem s x → Eq (f x) (g x)
                       hxs : Not (Membership.mem s x)
                       ⊢ Eq (s.mulIndicator f x) (t.mulIndicator g x)
                     -/
      (fun hxs => by simp only [mulIndicator_of_not_mem, hxs, mt hst.2 hxs, not_false_eq_true])
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem mulIndicator_union_eventuallyEq (h : ∀ᶠ a in l, a ∉ s ∩ t) :
    mulIndicator (s ∪ t) f =ᶠ[l] mulIndicator s f * mulIndicator t f :=
  h.mono fun _a ha => mulIndicator_union_of_not_mem_inter ha _


@[to_additive]
theorem mulIndicator_eventuallyLE_mulIndicator (h : f ≤ᶠ[l ⊓ 𝓟 s] g) :
    mulIndicator s f ≤ᶠ[l] mulIndicator s g :=
  (eventually_inf_principal.1 h).mono fun _ => mulIndicator_rel_mulIndicator le_rfl


@[to_additive]
theorem Monotone.mulIndicator_eventuallyEq_iUnion {ι} [Preorder ι] [One β] (s : ι → Set α)
    (hs : Monotone s) (f : α → β) (a : α) :
    (fun i => mulIndicator (s i) f a) =ᶠ[atTop] fun _ ↦ mulIndicator (⋃ i, s i) f a := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : One β
    s : ι → Set α
    hs : Monotone s
    f : α → β
    a : α
    ⊢ Filter.atTop.EventuallyEq (fun i => (s i).mulIndicator f a) fun x => (Set.iU …
  -/
  classical exact hs.piecewise_eventually_eq_iUnion f 1 a
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Monotone.tendsto_mulIndicator {ι} [Preorder ι] [One β] (s : ι → Set α) (hs : Monotone s)
    (f : α → β) (a : α) :
    Tendsto (fun i => mulIndicator (s i) f a) atTop (pure <| mulIndicator (⋃ i, s i) f a) :=
  tendsto_pure.2 <| hs.mulIndicator_eventuallyEq_iUnion s f a


@[to_additive]
theorem Antitone.mulIndicator_eventuallyEq_iInter {ι} [Preorder ι] [One β] (s : ι → Set α)
    (hs : Antitone s) (f : α → β) (a : α) :
    (fun i => mulIndicator (s i) f a) =ᶠ[atTop] fun _ ↦ mulIndicator (⋂ i, s i) f a := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : One β
    s : ι → Set α
    hs : Antitone s
    f : α → β
    a : α
    ⊢ Filter.atTop.EventuallyEq (fun i => (s i).mulIndicator f a) fun x => (Set.iI …
  -/
  classical exact hs.piecewise_eventually_eq_iInter f 1 a
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Antitone.tendsto_mulIndicator {ι} [Preorder ι] [One β] (s : ι → Set α) (hs : Antitone s)
    (f : α → β) (a : α) :
    Tendsto (fun i => mulIndicator (s i) f a) atTop (pure <| mulIndicator (⋂ i, s i) f a) :=
  tendsto_pure.2 <| hs.mulIndicator_eventuallyEq_iInter s f a


@[to_additive]
theorem mulIndicator_biUnion_finset_eventuallyEq {ι} [One β] (s : ι → Set α) (f : α → β) (a : α) :
    (fun n : Finset ι => mulIndicator (⋃ i ∈ n, s i) f a) =ᶠ[atTop]
      fun _ ↦ mulIndicator (iUnion s) f a := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    inst✝ : One β
    s : ι → Set α
    f : α → β
    a : α
    ⊢ Filter.atTop.EventuallyEq (fun n => (Set.iUnion fun i => Set.iUnion fun h => …
  -/
  rw [iUnion_eq_iUnion_finset s]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    inst✝ : One β
    s : ι → Set α
    f : α → β
    a : α
    ⊢ Filter.atTop.EventuallyEq (fun n => (Set.iUnion fun i => Set.iUnion fun h => …
  -/
  apply Monotone.mulIndicator_eventuallyEq_iUnion
  /-
    case hs
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    inst✝ : One β
    s : ι → Set α
    f : α → β
    a : α
    ⊢ Monotone fun i => Set.iUnion fun i_1 => Set.iUnion fun h => s i_1
  -/
  exact fun _ _ ↦ biUnion_subset_biUnion_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tendsto_mulIndicator_biUnion_finset {ι} [One β] (s : ι → Set α) (f : α → β) (a : α) :
    Tendsto (fun n : Finset ι => mulIndicator (⋃ i ∈ n, s i) f a) atTop
      (pure <| mulIndicator (iUnion s) f a) :=
  tendsto_pure.2 <| mulIndicator_biUnion_finset_eventuallyEq s f a


@[to_additive]
protected theorem Filter.EventuallyEq.mulSupport [One β] {f g : α → β} {l : Filter α}
    (h : f =ᶠ[l] g) :
    Function.mulSupport f =ᶠ[l] Function.mulSupport g :=
  h.preimage ({1}ᶜ : Set β)


@[to_additive]
protected theorem Filter.EventuallyEq.mulIndicator [One β] {l : Filter α} {f g : α → β} {s : Set α}
    (hfg : f =ᶠ[l] g) : s.mulIndicator f =ᶠ[l] s.mulIndicator g :=
  mulIndicator_eventuallyEq (hfg.filter_mono inf_le_left) EventuallyEq.rfl


@[to_additive]
theorem Filter.EventuallyEq.mulIndicator_one [One β] {l : Filter α} {f : α → β} {s : Set α}
    (hf : f =ᶠ[l] 1) : s.mulIndicator f =ᶠ[l] 1 :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                inst✝ : One β
                                l : Filter α
                                f : α → β
                                s : Set α
                                hf : l.EventuallyEq f 1
                                ⊢ l.EventuallyEq (s.mulIndicator 1) 1
                              -/
  hf.mulIndicator.trans <| by rw [mulIndicator_one']
                              /-
                                🎉 no goals
                              -/


@[to_additive]
theorem Filter.EventuallyEq.of_mulIndicator [One β] {l : Filter α} {f : α → β}
    (hf : ∀ᶠ x in l, f x ≠ 1) {s t : Set α} (h : s.mulIndicator f =ᶠ[l] t.mulIndicator f) :
    s =ᶠ[l] t := by
  have : ∀ {s : Set α}, Function.mulSupport (s.mulIndicator f) =ᶠ[l] s := fun {s} ↦ by
    rw [mulSupport_mulIndicator]
    exact (hf.mono fun x hx ↦ and_iff_left hx).set_eq
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : One β
    l : Filter α
    f : α → β
    hf : Filter.Eventually (fun x => Ne (f x) 1) l
    s t : Set α
    h : l.EventuallyEq (s.mulIndicator f) (t.mulIndicator f)
    this : ∀ {s : Set α}, l.EventuallyEq (Function.mulSupport (s.mulIndicator f)) s
    ⊢ l.EventuallyEq s t
  -/
  exact this.symm.trans <| h.mulSupport.trans this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Filter.EventuallyEq.of_mulIndicator_const [One β] {l : Filter α} {c : β} (hc : c ≠ 1)
    {s t : Set α} (h : s.mulIndicator (fun _ ↦ c) =ᶠ[l] t.mulIndicator fun _ ↦ c) : s =ᶠ[l] t :=
  .of_mulIndicator (Eventually.of_forall fun _ ↦ hc) h


@[to_additive]
theorem Filter.mulIndicator_const_eventuallyEq [One β] {l : Filter α} {c : β} (hc : c ≠ 1)
    {s t : Set α} : s.mulIndicator (fun _ ↦ c) =ᶠ[l] t.mulIndicator (fun _ ↦ c) ↔ s =ᶠ[l] t :=
  ⟨.of_mulIndicator_const hc, mulIndicator_eventuallyEq .rfl⟩

