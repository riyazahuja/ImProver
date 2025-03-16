lemma ker_def (f : Filter α) : f.ker = ⋂ s ∈ f, s := sInter_eq_biInter


@[simp] lemma mem_ker : a ∈ f.ker ↔ ∀ s ∈ f, a ∈ s := mem_sInter

@[simp] lemma subset_ker : s ⊆ f.ker ↔ ∀ t ∈ f, s ⊆ t := subset_sInter_iff


/-- `Filter.principal` forms a Galois coinsertion with `Filter.ker`. -/
def gi_principal_ker : GaloisCoinsertion (𝓟 : Set α → Filter α) ker :=
                                                     /-
                                                       ι : Sort u_1
                                                       α : Type u_2
                                                       β : Type u_3
                                                       f✝ g : Filter α
                                                       s✝ : Set α
                                                       a : α
                                                       s : Set α
                                                       f : Filter α
                                                       ⊢ Iff (LE.le (Filter.principal s) f) (LE.le s f.ker)
                                                     -/
  GaloisConnection.toGaloisCoinsertion (fun s f ↦ by simp [principal_le_iff]) <| by
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      f g : Filter α
      s : Set α
      a : α
      ⊢ ∀ (a : Set α), LE.le (Filter.principal a).ker a
    -/
    simp only [le_iff_subset, subset_def, mem_ker, mem_principal]; aesop
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma ker_mono : Monotone (ker : Filter α → Set α) := gi_principal_ker.gc.monotone_u

lemma ker_surjective : Surjective (ker : Filter α → Set α) := gi_principal_ker.u_surjective


@[simp] lemma ker_bot : ker (⊥ : Filter α) = ∅ := sInter_eq_empty_iff.2 fun _ ↦ ⟨∅, trivial, id⟩

@[simp] lemma ker_top : ker (⊤ : Filter α) = univ := gi_principal_ker.gc.u_top

                                                                                             /-
                                                                                               α : Type u_2
                                                                                               f : Filter α
                                                                                               ⊢ Iff (LE.le (Filter.principal Top.top) f) (Eq f Top.top)
                                                                                             -/
@[simp] lemma ker_eq_univ : ker f = univ ↔ f = ⊤ := gi_principal_ker.gc.u_eq_top.trans <| by simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/

@[simp] lemma ker_inf (f g : Filter α) : ker (f ⊓ g) = ker f ∩ ker g := gi_principal_ker.gc.u_inf

@[simp] lemma ker_iInf (f : ι → Filter α) : ker (⨅ i, f i) = ⋂ i, ker (f i) :=
  gi_principal_ker.gc.u_iInf

@[simp] lemma ker_sInf (S : Set (Filter α)) : ker (sInf S) = ⋂ f ∈ S, ker f :=
  gi_principal_ker.gc.u_sInf

@[simp] lemma ker_principal (s : Set α) : ker (𝓟 s) = s := gi_principal_ker.u_l_eq _


                                                          /-
                                                            α : Type u_2
                                                            a : α
                                                            ⊢ Eq (Pure.pure a).ker (Singleton.singleton a)
                                                          -/
@[simp] lemma ker_pure (a : α) : ker (pure a) = {a} := by rw [← principal_singleton, ker_principal]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp] lemma ker_comap (m : α → β) (f : Filter β) : ker (comap m f) = m ⁻¹' ker f := by
  /-
    α : Type u_2
    β : Type u_3
    m : α → β
    f : Filter β
    ⊢ Eq (Filter.comap m f).ker (Set.preimage m f.ker)
  -/
  ext a
  /-
    case h
    α : Type u_2
    β : Type u_3
    m : α → β
    f : Filter β
    a : α
    ⊢ Iff (Membership.mem (Filter.comap m f).ker a) (Membership.mem (Set.preimage  …
  -/
  simp only [mem_ker, mem_comap, forall_exists_index, and_imp, @forall_swap (Set α), mem_preimage]
  /-
    case h
    α : Type u_2
    β : Type u_3
    m : α → β
    f : Filter β
    a : α
    ⊢ Iff (∀ (y : Set β), Membership.mem f y → ∀ (x : Set α), HasSubset.Subset (Se …
  -/
  exact forall₂_congr fun s _ ↦ ⟨fun h ↦ h _ Subset.rfl, fun ha t ht ↦ ht ha⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_iSup (f : ι → Filter α) : ker (⨆ i, f i) = ⋃ i, ker (f i) := by
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → Filter α
    ⊢ Eq (iSup fun i => f i).ker (Set.iUnion fun i => (f i).ker)
  -/
  refine subset_antisymm (fun x hx ↦ ?_) ker_mono.le_map_iSup
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → Filter α
    x : α
    hx : Membership.mem (iSup fun i => f i).ker x
    ⊢ Membership.mem (Set.iUnion fun i => (f i).ker) x
  -/
  simp only [mem_iUnion, mem_ker] at hx ⊢
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → Filter α
    x : α
    hx : ∀ (s : Set α), Membership.mem (iSup fun i => f i) s → Membership.mem s x
    ⊢ Exists fun i => ∀ (s : Set α), Membership.mem (f i) s → Membership.mem s x
  -/
  contrapose! hx
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → Filter α
    x : α
    hx : ∀ (i : ι), Exists fun s => And (Membership.mem (f i) s) (Not (Membership. …
    ⊢ Exists fun s => And (Membership.mem (iSup fun i => f i) s) (Not (Membership. …
  -/
  choose s hsf hxs using hx
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → Filter α
    x : α
    s : ι → Set α
    hsf : ∀ (i : ι), Membership.mem (f i) (s i)
    hxs : ∀ (i : ι), Not (Membership.mem (s i) x)
    ⊢ Exists fun s => And (Membership.mem (iSup fun i => f i) s) (Not (Membership. …
  -/
  refine ⟨⋃ i, s i, ?_, by simpa⟩
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → Filter α
    x : α
    s : ι → Set α
    hsf : ∀ (i : ι), Membership.mem (f i) (s i)
    hxs : ∀ (i : ι), Not (Membership.mem (s i) x)
    ⊢ Membership.mem (iSup fun i => f i) (Set.iUnion fun i => s i)
  -/
  exact mem_iSup.2 fun i ↦ mem_of_superset (hsf i) (subset_iUnion s i)
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_sSup (S : Set (Filter α)) : ker (sSup S) = ⋃ f ∈ S, ker f := by
  /-
    α : Type u_2
    S : Set (Filter α)
    ⊢ Eq (SupSet.sSup S).ker (Set.iUnion fun f => Set.iUnion fun h => f.ker)
  -/
  simp [sSup_eq_iSup]
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_sup (f g : Filter α) : ker (f ⊔ g) = ker f ∪ ker g := by
  /-
    α : Type u_2
    f g : Filter α
    ⊢ Eq (Max.max f g).ker (Union.union f.ker g.ker)
  -/
  rw [← sSup_pair, ker_sSup, biUnion_pair]
  /-
    🎉 no goals
  -/


