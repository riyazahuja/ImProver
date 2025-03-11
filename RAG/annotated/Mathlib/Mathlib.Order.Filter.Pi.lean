theorem tendsto_eval_pi (f : ∀ i, Filter (α i)) (i : ι) : Tendsto (eval i) (pi f) (f i) :=
  tendsto_iInf' i tendsto_comap


theorem tendsto_pi {β : Type*} {m : β → ∀ i, α i} {l : Filter β} :
    Tendsto m l (pi f) ↔ ∀ i, Tendsto (fun x => m x i) l (f i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    β : Type u_3
    m : β → (i : ι) → α i
    l : Filter β
    ⊢ Iff (Filter.Tendsto m l (Filter.pi f)) (∀ (i : ι), Filter.Tendsto (fun x =>  …
  -/
  simp only [pi, tendsto_iInf, tendsto_comap_iff]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If a function tends to a product `Filter.pi f` of filters, then its `i`-th component tends to
`f i`. See also `Filter.Tendsto.apply_nhds` for the special case of converging to a point in a
product of topological spaces. -/
alias ⟨Tendsto.apply, _⟩ := tendsto_pi


theorem le_pi {g : Filter (∀ i, α i)} : g ≤ pi f ↔ ∀ i, Tendsto (eval i) g (f i) :=
  tendsto_pi


@[mono]
theorem pi_mono (h : ∀ i, f₁ i ≤ f₂ i) : pi f₁ ≤ pi f₂ :=
  iInf_mono fun i => comap_mono <| h i


theorem mem_pi_of_mem (i : ι) {s : Set (α i)} (hs : s ∈ f i) : eval i ⁻¹' s ∈ pi f :=
  mem_iInf_of_mem i <| preimage_mem_comap hs


theorem pi_mem_pi {I : Set ι} (hI : I.Finite) (h : ∀ i ∈ I, s i ∈ f i) : I.pi s ∈ pi f := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    s : (i : ι) → Set (α i)
    I : Set ι
    hI : I.Finite
    h : ∀ (i : ι), Membership.mem I i → Membership.mem (f i) (s i)
    ⊢ Membership.mem (Filter.pi f) (I.pi s)
  -/
  rw [pi_def, biInter_eq_iInter]
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    s : (i : ι) → Set (α i)
    I : Set ι
    hI : I.Finite
    h : ∀ (i : ι), Membership.mem I i → Membership.mem (f i) (s i)
    ⊢ Membership.mem (Filter.pi f) (Set.iInter fun x => Set.preimage (Function.eva …
  -/
  refine mem_iInf_of_iInter hI (fun i => ?_) Subset.rfl
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    s : (i : ι) → Set (α i)
    I : Set ι
    hI : I.Finite
    h : ∀ (i : ι), Membership.mem I i → Membership.mem (f i) (s i)
    i : ↑I
    ⊢ Membership.mem (Filter.comap (Function.eval ↑i) (f ↑i)) (Set.preimage (Funct …
  -/
  exact preimage_mem_comap (h i i.2)
  /-
    🎉 no goals
  -/


theorem mem_pi {s : Set (∀ i, α i)} :
    s ∈ pi f ↔ ∃ I : Set ι, I.Finite ∧ ∃ t : ∀ i, Set (α i), (∀ i, t i ∈ f i) ∧ I.pi t ⊆ s := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    s : Set ((i : ι) → α i)
    ⊢ Iff (Membership.mem (Filter.pi f) s) (Exists fun I => And I.Finite (Exists f …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : Set ((i : ι) → α i)
      ⊢ Membership.mem (Filter.pi f) s → Exists fun I => And I.Finite (Exists fun t  …
    -/
  · simp only [pi, mem_iInf', mem_comap, pi_def]
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : Set ((i : ι) → α i)
      ⊢ (Exists fun I => And I.Finite (Exists fun V => And (∀ (i : ι), Exists fun t  …
    -/
    rintro ⟨I, If, V, hVf, -, rfl, -⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      I : Set ι
      If : I.Finite
      V : ι → Set ((i : ι) → α i)
      hVf : ∀ (i : ι), Exists fun t => And (Membership.mem (f i) t) (HasSubset.Subse …
      ⊢ Exists fun I_1 => And I_1.Finite (Exists fun t => And (∀ (i : ι), Membership …
    -/
    choose t htf htV using hVf
    /-
      case mp.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      I : Set ι
      If : I.Finite
      V : ι → Set ((i : ι) → α i)
      t : (i : ι) → Set (α i)
      htf : ∀ (i : ι), Membership.mem (f i) (t i)
      htV : ∀ (i : ι), HasSubset.Subset (Set.preimage (Function.eval i) (t i)) (V i)
      ⊢ Exists fun I_1 => And I_1.Finite (Exists fun t => And (∀ (i : ι), Membership …
    -/
    exact ⟨I, If, t, htf, iInter₂_mono fun i _ => htV i⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : Set ((i : ι) → α i)
      ⊢ (Exists fun I => And I.Finite (Exists fun t => And (∀ (i : ι), Membership.me …
    -/
  · rintro ⟨I, If, t, htf, hts⟩
    /-
      case mpr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : Set ((i : ι) → α i)
      I : Set ι
      If : I.Finite
      t : (i : ι) → Set (α i)
      htf : ∀ (i : ι), Membership.mem (f i) (t i)
      hts : HasSubset.Subset (I.pi t) s
      ⊢ Membership.mem (Filter.pi f) s
    -/
    exact mem_of_superset (pi_mem_pi If fun i _ => htf i) hts
    /-
      🎉 no goals
    -/


theorem mem_pi' {s : Set (∀ i, α i)} :
    s ∈ pi f ↔ ∃ I : Finset ι, ∃ t : ∀ i, Set (α i), (∀ i, t i ∈ f i) ∧ Set.pi (↑I) t ⊆ s :=
  mem_pi.trans exists_finite_iff_finset


theorem mem_of_pi_mem_pi [∀ i, NeBot (f i)] {I : Set ι} (h : I.pi s ∈ pi f) {i : ι} (hi : i ∈ I) :
    s i ∈ f i := by
  classical
  rcases mem_pi.1 h with ⟨I', -, t, htf, hts⟩
  refine mem_of_superset (htf i) fun x hx => ?_
  have : ∀ i, (t i).Nonempty := fun i => nonempty_of_mem (htf i)
  choose g hg using this
  have : update g i x ∈ I'.pi t := fun j _ => by
    rcases eq_or_ne j i with (rfl | hne) <;> simp [*]
  simpa using hts this i hi


@[simp]
theorem pi_mem_pi_iff [∀ i, NeBot (f i)] {I : Set ι} (hI : I.Finite) :
    I.pi s ∈ pi f ↔ ∀ i ∈ I, s i ∈ f i :=
  ⟨fun h _i hi => mem_of_pi_mem_pi h hi, pi_mem_pi hI⟩


theorem Eventually.eval_pi {i : ι} (hf : ∀ᶠ x : α i in f i, p i x) :
    ∀ᶠ x : ∀ i : ι, α i in pi f, p i (x i) := (tendsto_eval_pi _ _).eventually hf


theorem eventually_pi [Finite ι] (hf : ∀ i, ∀ᶠ x in f i, p i x) :
    ∀ᶠ x : ∀ i, α i in pi f, ∀ i, p i (x i) := eventually_all.2 fun _i => (hf _).eval_pi


theorem hasBasis_pi {ι' : ι → Type*} {s : ∀ i, ι' i → Set (α i)} {p : ∀ i, ι' i → Prop}
    (h : ∀ i, (f i).HasBasis (p i) (s i)) :
    (pi f).HasBasis (fun If : Set ι × ∀ i, ι' i => If.1.Finite ∧ ∀ i ∈ If.1, p i (If.2 i))
      fun If : Set ι × ∀ i, ι' i => If.1.pi fun i => s i <| If.2 i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    ι' : ι → Type u_3
    s : (i : ι) → ι' i → Set (α i)
    p : (i : ι) → ι' i → Prop
    h : ∀ (i : ι), (f i).HasBasis (p i) (s i)
    ⊢ (Filter.pi f).HasBasis (fun If => And If.1.Finite (∀ (i : ι), Membership.mem …
  -/
  simpa [Set.pi_def] using hasBasis_iInf' fun i => (h i).comap (eval i : (∀ j, α j) → α i)
  /-
    🎉 no goals
  -/


theorem le_pi_principal (s : (i : ι) → Set (α i)) :
    𝓟 (univ.pi s) ≤ pi fun i ↦ 𝓟 (s i) :=
  le_pi.2 fun i ↦ tendsto_principal_principal.2 fun _f hf ↦ hf i trivial


@[simp]
theorem pi_principal [Finite ι] (s : (i : ι) → Set (α i)) :
    pi (fun i ↦ 𝓟 (s i)) = 𝓟 (univ.pi s) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : Finite ι
    s : (i : ι) → Set (α i)
    ⊢ Eq (Filter.pi fun i => Filter.principal (s i)) (Filter.principal (Set.univ.p …
  -/
  simp [Filter.pi, Set.pi_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem pi_pure [Finite ι] (f : (i : ι) → α i) : pi (pure <| f ·) = pure f := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : Finite ι
    f : (i : ι) → α i
    ⊢ Eq (Filter.pi fun x => Pure.pure (f x)) (Pure.pure f)
  -/
  simp only [← principal_singleton, pi_principal, univ_pi_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem pi_inf_principal_univ_pi_eq_bot :
    pi f ⊓ 𝓟 (Set.pi univ s) = ⊥ ↔ ∃ i, f i ⊓ 𝓟 (s i) = ⊥ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    s : (i : ι) → Set (α i)
    ⊢ Iff (Eq (Min.min (Filter.pi f) (Filter.principal (Set.univ.pi s))) Bot.bot)  …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      ⊢ Eq (Min.min (Filter.pi f) (Filter.principal (Set.univ.pi s))) Bot.bot → Exis …
    -/
  · simp only [inf_principal_eq_bot, mem_pi]
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      ⊢ (Exists fun I => And I.Finite (Exists fun t => And (∀ (i : ι), Membership.me …
    -/
    contrapose!
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      ⊢ (∀ (i : ι), Not (Membership.mem (f i) (HasCompl.compl (s i)))) → ∀ (I : Set  …
    -/
    rintro (hsf : ∀ i, ∃ᶠ x in f i, x ∈ s i) I - t htf hts
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      hsf : ∀ (i : ι), Filter.Frequently (fun x => Membership.mem (s i) x) (f i)
      I : Set ι
      t : (i : ι) → Set (α i)
      htf : ∀ (i : ι), Membership.mem (f i) (t i)
      hts : HasSubset.Subset (I.pi t) (HasCompl.compl (Set.univ.pi s))
      ⊢ False
    -/
    have : ∀ i, (s i ∩ t i).Nonempty := fun i => ((hsf i).and_eventually (htf i)).exists
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      hsf : ∀ (i : ι), Filter.Frequently (fun x => Membership.mem (s i) x) (f i)
      I : Set ι
      t : (i : ι) → Set (α i)
      htf : ∀ (i : ι), Membership.mem (f i) (t i)
      hts : HasSubset.Subset (I.pi t) (HasCompl.compl (Set.univ.pi s))
      this : ∀ (i : ι), (Inter.inter (s i) (t i)).Nonempty
      ⊢ False
    -/
    choose x hxs hxt using this
    /-
      case mp
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      hsf : ∀ (i : ι), Filter.Frequently (fun x => Membership.mem (s i) x) (f i)
      I : Set ι
      t : (i : ι) → Set (α i)
      htf : ∀ (i : ι), Membership.mem (f i) (t i)
      hts : HasSubset.Subset (I.pi t) (HasCompl.compl (Set.univ.pi s))
      x : (i : ι) → α i
      hxs : ∀ (i : ι), Membership.mem (s i) (x i)
      hxt : ∀ (i : ι), Membership.mem (t i) (x i)
      ⊢ False
    -/
    exact hts (fun i _ => hxt i) (mem_univ_pi.2 hxs)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      ⊢ (Exists fun i => Eq (Min.min (f i) (Filter.principal (s i))) Bot.bot) → Eq ( …
    -/
  · simp only [inf_principal_eq_bot]
    /-
      case mpr
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      ⊢ (Exists fun i => Membership.mem (f i) (HasCompl.compl (s i))) → Membership.m …
    -/
    rintro ⟨i, hi⟩
    /-
      case mpr.intro
      ι : Type u_1
      α : ι → Type u_2
      f : (i : ι) → Filter (α i)
      s : (i : ι) → Set (α i)
      i : ι
      hi : Membership.mem (f i) (HasCompl.compl (s i))
      ⊢ Membership.mem (Filter.pi f) (HasCompl.compl (Set.univ.pi s))
    -/
    filter_upwards [mem_pi_of_mem i hi] with x using mt fun h => h i trivial
    /-
      🎉 no goals
    -/


@[simp]
theorem pi_inf_principal_pi_eq_bot [∀ i, NeBot (f i)] {I : Set ι} :
    pi f ⊓ 𝓟 (Set.pi I s) = ⊥ ↔ ∃ i ∈ I, f i ⊓ 𝓟 (s i) = ⊥ := by
  classical
  rw [← univ_pi_piecewise_univ I, pi_inf_principal_univ_pi_eq_bot]
  refine exists_congr fun i => ?_
  by_cases hi : i ∈ I <;> simp [hi, NeBot.ne']


@[simp]
theorem pi_inf_principal_univ_pi_neBot :
                                                                        /-
                                                                          ι : Type u_1
                                                                          α : ι → Type u_2
                                                                          f : (i : ι) → Filter (α i)
                                                                          s : (i : ι) → Set (α i)
                                                                          ⊢ Iff (Min.min (Filter.pi f) (Filter.principal (Set.univ.pi s))).NeBot (∀ (i : …
                                                                        -/
    NeBot (pi f ⊓ 𝓟 (Set.pi univ s)) ↔ ∀ i, NeBot (f i ⊓ 𝓟 (s i)) := by simp [neBot_iff]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem pi_inf_principal_pi_neBot [∀ i, NeBot (f i)] {I : Set ι} :
                                                                     /-
                                                                       ι : Type u_1
                                                                       α : ι → Type u_2
                                                                       f : (i : ι) → Filter (α i)
                                                                       s : (i : ι) → Set (α i)
                                                                       inst✝ : ∀ (i : ι), (f i).NeBot
                                                                       I : Set ι
                                                                       ⊢ Iff (Min.min (Filter.pi f) (Filter.principal (I.pi s))).NeBot (∀ (i : ι), Me …
                                                                     -/
    NeBot (pi f ⊓ 𝓟 (I.pi s)) ↔ ∀ i ∈ I, NeBot (f i ⊓ 𝓟 (s i)) := by simp [neBot_iff]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance PiInfPrincipalPi.neBot [h : ∀ i, NeBot (f i ⊓ 𝓟 (s i))] {I : Set ι} :
    NeBot (pi f ⊓ 𝓟 (I.pi s)) :=
  (pi_inf_principal_univ_pi_neBot.2 ‹_›).mono <|
    inf_le_inf_left _ <| principal_mono.2 fun _ hx i _ => hx i trivial


@[simp]
theorem pi_eq_bot : pi f = ⊥ ↔ ∃ i, f i = ⊥ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    ⊢ Iff (Eq (Filter.pi f) Bot.bot) (Exists fun i => Eq (f i) Bot.bot)
  -/
  simpa using @pi_inf_principal_univ_pi_eq_bot ι α f fun _ => univ
  /-
    🎉 no goals
  -/


@[simp]
                                                         /-
                                                           ι : Type u_1
                                                           α : ι → Type u_2
                                                           f : (i : ι) → Filter (α i)
                                                           ⊢ Iff (Filter.pi f).NeBot (∀ (i : ι), (f i).NeBot)
                                                         -/
theorem pi_neBot : NeBot (pi f) ↔ ∀ i, NeBot (f i) := by simp [neBot_iff]
                                                         /-
                                                           🎉 no goals
                                                         -/


instance [∀ i, NeBot (f i)] : NeBot (pi f) :=
  pi_neBot.2 ‹_›


@[simp]
theorem map_eval_pi (f : ∀ i, Filter (α i)) [∀ i, NeBot (f i)] (i : ι) :
    map (eval i) (pi f) = f i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f i).NeBot
    i : ι
    ⊢ Eq (Filter.map (Function.eval i) (Filter.pi f)) (f i)
  -/
  refine le_antisymm (tendsto_eval_pi f i) fun s hs => ?_
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f i).NeBot
    i : ι
    s : Set (α i)
    hs : Membership.mem (Filter.map (Function.eval i) (Filter.pi f)) s
    ⊢ Membership.mem (f i) s
  -/
  rcases mem_pi.1 (mem_map.1 hs) with ⟨I, hIf, t, htf, hI⟩
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f i).NeBot
    i : ι
    s : Set (α i)
    hs : Membership.mem (Filter.map (Function.eval i) (Filter.pi f)) s
    I : Set ι
    hIf : I.Finite
    t : (i : ι) → Set (α i)
    htf : ∀ (i : ι), Membership.mem (f i) (t i)
    hI : HasSubset.Subset (I.pi t) (Set.preimage (Function.eval i) s)
    ⊢ Membership.mem (f i) s
  -/
  rw [← image_subset_iff] at hI
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f i).NeBot
    i : ι
    s : Set (α i)
    hs : Membership.mem (Filter.map (Function.eval i) (Filter.pi f)) s
    I : Set ι
    hIf : I.Finite
    t : (i : ι) → Set (α i)
    htf : ∀ (i : ι), Membership.mem (f i) (t i)
    hI : HasSubset.Subset (Set.image (Function.eval i) (I.pi t)) s
    ⊢ Membership.mem (f i) s
  -/
  refine mem_of_superset (htf i) ((subset_eval_image_pi ?_ _).trans hI)
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f i).NeBot
    i : ι
    s : Set (α i)
    hs : Membership.mem (Filter.map (Function.eval i) (Filter.pi f)) s
    I : Set ι
    hIf : I.Finite
    t : (i : ι) → Set (α i)
    htf : ∀ (i : ι), Membership.mem (f i) (t i)
    hI : HasSubset.Subset (Set.image (Function.eval i) (I.pi t)) s
    ⊢ (I.pi t).Nonempty
  -/
  exact nonempty_of_mem (pi_mem_pi hIf fun i _ => htf i)
  /-
    🎉 no goals
  -/


@[simp]
theorem pi_le_pi [∀ i, NeBot (f₁ i)] : pi f₁ ≤ pi f₂ ↔ ∀ i, f₁ i ≤ f₂ i :=
  ⟨fun h i => map_eval_pi f₁ i ▸ (tendsto_eval_pi _ _).mono_left h, pi_mono⟩


@[simp]
theorem pi_inj [∀ i, NeBot (f₁ i)] : pi f₁ = pi f₂ ↔ f₁ = f₂ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f₁ f₂ : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f₁ i).NeBot
    ⊢ Iff (Eq (Filter.pi f₁) (Filter.pi f₂)) (Eq f₁ f₂)
  -/
  refine ⟨fun h => ?_, congr_arg pi⟩
  /-
    ι : Type u_1
    α : ι → Type u_2
    f₁ f₂ : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f₁ i).NeBot
    h : Eq (Filter.pi f₁) (Filter.pi f₂)
    ⊢ Eq f₁ f₂
  -/
  have hle : f₁ ≤ f₂ := pi_le_pi.1 h.le
  /-
    ι : Type u_1
    α : ι → Type u_2
    f₁ f₂ : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f₁ i).NeBot
    h : Eq (Filter.pi f₁) (Filter.pi f₂)
    hle : LE.le f₁ f₂
    ⊢ Eq f₁ f₂
  -/
  haveI : ∀ i, NeBot (f₂ i) := fun i => neBot_of_le (hle i)
  /-
    ι : Type u_1
    α : ι → Type u_2
    f₁ f₂ : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), (f₁ i).NeBot
    h : Eq (Filter.pi f₁) (Filter.pi f₂)
    hle : LE.le f₁ f₂
    this : ∀ (i : ι), (f₂ i).NeBot
    ⊢ Eq f₁ f₂
  -/
  exact hle.antisymm (pi_le_pi.1 h.ge)
  /-
    🎉 no goals
  -/


theorem tendsto_piMap_pi {β : ι → Type*} {f : ∀ i, α i → β i} {l : ∀ i, Filter (α i)}
    {l' : ∀ i, Filter (β i)} (h : ∀ i, Tendsto (f i) (l i) (l' i)) :
    Tendsto (Pi.map f) (pi l) (pi l') :=
  tendsto_pi.2 fun i ↦ (h i).comp (tendsto_eval_pi _ _)


/-- Coproduct of filters. -/
protected def coprodᵢ (f : ∀ i, Filter (α i)) : Filter (∀ i, α i) :=
  ⨆ i : ι, comap (eval i) (f i)


theorem mem_coprodᵢ_iff {s : Set (∀ i, α i)} :
                                                                        /-
                                                                          ι : Type u_1
                                                                          α : ι → Type u_2
                                                                          f : (i : ι) → Filter (α i)
                                                                          s : Set ((i : ι) → α i)
                                                                          ⊢ Iff (Membership.mem (Filter.coprodᵢ f) s) (∀ (i : ι), Exists fun t₁ => And ( …
                                                                        -/
    s ∈ Filter.coprodᵢ f ↔ ∀ i : ι, ∃ t₁ ∈ f i, eval i ⁻¹' t₁ ⊆ s := by simp [Filter.coprodᵢ]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem compl_mem_coprodᵢ {s : Set (∀ i, α i)} :
    sᶜ ∈ Filter.coprodᵢ f ↔ ∀ i, (eval i '' s)ᶜ ∈ f i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    s : Set ((i : ι) → α i)
    ⊢ Iff (Membership.mem (Filter.coprodᵢ f) (HasCompl.compl s)) (∀ (i : ι), Membe …
  -/
  simp only [Filter.coprodᵢ, mem_iSup, compl_mem_comap]
  /-
    🎉 no goals
  -/


theorem coprodᵢ_neBot_iff' :
    NeBot (Filter.coprodᵢ f) ↔ (∀ i, Nonempty (α i)) ∧ ∃ d, NeBot (f d) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    ⊢ Iff (Filter.coprodᵢ f).NeBot (And (∀ (i : ι), Nonempty (α i)) (Exists fun d  …
  -/
  simp only [Filter.coprodᵢ, iSup_neBot, ← exists_and_left, ← comap_eval_neBot_iff']
  /-
    🎉 no goals
  -/


@[simp]
theorem coprodᵢ_neBot_iff [∀ i, Nonempty (α i)] : NeBot (Filter.coprodᵢ f) ↔ ∃ d, NeBot (f d) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), Nonempty (α i)
    ⊢ Iff (Filter.coprodᵢ f).NeBot (Exists fun d => (f d).NeBot)
  -/
  simp [coprodᵢ_neBot_iff', *]
  /-
    🎉 no goals
  -/


theorem coprodᵢ_eq_bot_iff' : Filter.coprodᵢ f = ⊥ ↔ (∃ i, IsEmpty (α i)) ∨ f = ⊥ := by
  simpa only [not_neBot, not_and_or, funext_iff, not_forall, not_exists, not_nonempty_iff]
    using coprodᵢ_neBot_iff'.not


@[simp]
theorem coprodᵢ_eq_bot_iff [∀ i, Nonempty (α i)] : Filter.coprodᵢ f = ⊥ ↔ f = ⊥ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    inst✝ : ∀ (i : ι), Nonempty (α i)
    ⊢ Iff (Eq (Filter.coprodᵢ f) Bot.bot) (Eq f Bot.bot)
  -/
  simpa [funext_iff] using coprodᵢ_neBot_iff.not
  /-
    🎉 no goals
  -/


@[simp] theorem coprodᵢ_bot' : Filter.coprodᵢ (⊥ : ∀ i, Filter (α i)) = ⊥ :=
  coprodᵢ_eq_bot_iff'.2 (Or.inr rfl)


@[simp]
theorem coprodᵢ_bot : Filter.coprodᵢ (fun _ => ⊥ : ∀ i, Filter (α i)) = ⊥ :=
  coprodᵢ_bot'


theorem NeBot.coprodᵢ [∀ i, Nonempty (α i)] {i : ι} (h : NeBot (f i)) : NeBot (Filter.coprodᵢ f) :=
  coprodᵢ_neBot_iff.2 ⟨i, h⟩


@[instance]
theorem coprodᵢ_neBot [∀ i, Nonempty (α i)] [Nonempty ι] (f : ∀ i, Filter (α i))
    [H : ∀ i, NeBot (f i)] : NeBot (Filter.coprodᵢ f) :=
  (H (Classical.arbitrary ι)).coprodᵢ


@[mono]
theorem coprodᵢ_mono (hf : ∀ i, f₁ i ≤ f₂ i) : Filter.coprodᵢ f₁ ≤ Filter.coprodᵢ f₂ :=
  iSup_mono fun i => comap_mono (hf i)


theorem map_pi_map_coprodᵢ_le :
    map (fun k : ∀ i, α i => fun i => m i (k i)) (Filter.coprodᵢ f) ≤
      Filter.coprodᵢ fun i => map (m i) (f i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    β : ι → Type u_3
    m : (i : ι) → α i → β i
    ⊢ LE.le (Filter.map (fun k i => m i (k i)) (Filter.coprodᵢ f)) (Filter.coprodᵢ …
  -/
  simp only [le_def, mem_map, mem_coprodᵢ_iff]
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    β : ι → Type u_3
    m : (i : ι) → α i → β i
    ⊢ ∀ (x : Set ((i : ι) → β i)), (∀ (i : ι), Exists fun t₁ => And (Membership.me …
  -/
  intro s h i
  /-
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    β : ι → Type u_3
    m : (i : ι) → α i → β i
    s : Set ((i : ι) → β i)
    h : ∀ (i : ι), Exists fun t₁ => And (Membership.mem (f i) (Set.preimage (m i)  …
    i : ι
    ⊢ Exists fun t₁ => And (Membership.mem (f i) t₁) (HasSubset.Subset (Set.preima …
  -/
  obtain ⟨t, H, hH⟩ := h i
  /-
    case intro.intro
    ι : Type u_1
    α : ι → Type u_2
    f : (i : ι) → Filter (α i)
    β : ι → Type u_3
    m : (i : ι) → α i → β i
    s : Set ((i : ι) → β i)
    h : ∀ (i : ι), Exists fun t₁ => And (Membership.mem (f i) (Set.preimage (m i)  …
    i : ι
    t : Set (β i)
    H : Membership.mem (f i) (Set.preimage (m i) t)
    hH : HasSubset.Subset (Set.preimage (Function.eval i) t) s
    ⊢ Exists fun t₁ => And (Membership.mem (f i) t₁) (HasSubset.Subset (Set.preima …
  -/
  exact ⟨{ x : α i | m i x ∈ t }, H, fun x hx => hH hx⟩
  /-
    🎉 no goals
  -/


theorem Tendsto.pi_map_coprodᵢ {g : ∀ i, Filter (β i)} (h : ∀ i, Tendsto (m i) (f i) (g i)) :
    Tendsto (fun k : ∀ i, α i => fun i => m i (k i)) (Filter.coprodᵢ f) (Filter.coprodᵢ g) :=
  map_pi_map_coprodᵢ_le.trans (coprodᵢ_mono h)


