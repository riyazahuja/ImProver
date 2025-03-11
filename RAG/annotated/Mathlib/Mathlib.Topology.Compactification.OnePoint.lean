/-- The OnePoint extension of an arbitrary topological space `X` -/
def OnePoint (X : Type*) :=
  Option X


/-- The repr uses the notation from the `OnePoint` locale. -/
instance [Repr X] : Repr (OnePoint X) :=
  ⟨fun o _ =>
    match o with
    | none => "∞"
    | some a => "↑" ++ repr a⟩


/-- The point at infinity -/
@[match_pattern] def infty : OnePoint X := none


@[inherit_doc]
scoped notation "∞" => OnePoint.infty


/-- Coercion from `X` to `OnePoint X`. -/
@[coe, match_pattern] def some : X → OnePoint X := Option.some


@[simp]
lemma some_eq_iff (x₁ x₂ : X) : (some x₁ = some x₂) ↔ (x₁ = x₂) := by
  /-
    X : Type u_1
    x₁ x₂ : X
    ⊢ Iff (Eq ↑x₁ ↑x₂) (Eq x₁ x₂)
  -/
  rw [iff_eq_eq]
  /-
    X : Type u_1
    x₁ x₂ : X
    ⊢ Eq (Eq ↑x₁ ↑x₂) (Eq x₁ x₂)
  -/
  exact Option.some.injEq x₁ x₂
  /-
    🎉 no goals
  -/


instance : CoeTC X (OnePoint X) := ⟨some⟩


instance : Inhabited (OnePoint X) := ⟨∞⟩


protected lemma «forall» {p : OnePoint X → Prop} :
    (∀ (x : OnePoint X), p x) ↔ p ∞ ∧ ∀ (x : X), p x :=
  Option.forall


protected lemma «exists» {p : OnePoint X → Prop} :
    (∃ x, p x) ↔ p ∞ ∨ ∃ (x : X), p x :=
  Option.exists


instance [Fintype X] : Fintype (OnePoint X) :=
  inferInstanceAs (Fintype (Option X))


instance infinite [Infinite X] : Infinite (OnePoint X) :=
  inferInstanceAs (Infinite (Option X))


theorem coe_injective : Function.Injective ((↑) : X → OnePoint X) :=
  Option.some_injective X


@[norm_cast]
theorem coe_eq_coe {x y : X} : (x : OnePoint X) = y ↔ x = y :=
  coe_injective.eq_iff


@[simp]
theorem coe_ne_infty (x : X) : (x : OnePoint X) ≠ ∞ :=
  nofun


@[simp]
theorem infty_ne_coe (x : X) : ∞ ≠ (x : OnePoint X) :=
  nofun


/-- Recursor for `OnePoint` using the preferred forms `∞` and `↑x`. -/
@[elab_as_elim, induction_eliminator, cases_eliminator]
protected def rec {C : OnePoint X → Sort*} (infty : C ∞) (coe : ∀ x : X, C x) :
    ∀ z : OnePoint X, C z
  | ∞ => infty
  | (x : X) => coe x


/-- An elimination principle for `OnePoint`. -/
@[inline] protected def elim : OnePoint X → Y → (X → Y) → Y := Option.elim


@[simp] theorem elim_infty (y : Y) (f : X → Y) : ∞.elim y f = y := rfl


@[simp] theorem elim_some (y : Y) (f : X → Y) (x : X) : (some x).elim y f = f x := rfl


theorem isCompl_range_coe_infty : IsCompl (range ((↑) : X → OnePoint X)) {∞} :=
  isCompl_range_some_none X

-- Porting note: moved @[simp] to a new lemma

theorem range_coe_union_infty : range ((↑) : X → OnePoint X) ∪ {∞} = univ :=
  range_some_union_none X


@[simp]
theorem insert_infty_range_coe : insert ∞ (range (@some X)) = univ :=
  insert_none_range_some _


@[simp]
theorem range_coe_inter_infty : range ((↑) : X → OnePoint X) ∩ {∞} = ∅ :=
  range_some_inter_none X


@[simp]
theorem compl_range_coe : (range ((↑) : X → OnePoint X))ᶜ = {∞} :=
  compl_range_some X


theorem compl_infty : ({∞}ᶜ : Set (OnePoint X)) = range ((↑) : X → OnePoint X) :=
  (@isCompl_range_coe_infty X).symm.compl_eq


theorem compl_image_coe (s : Set X) : ((↑) '' s : Set (OnePoint X))ᶜ = (↑) '' sᶜ ∪ {∞} := by
  /-
    X : Type u_1
    s : Set X
    ⊢ Eq (HasCompl.compl (Set.image OnePoint.some s)) (Union.union (Set.image OneP …
  -/
  rw [coe_injective.compl_image_eq, compl_range_coe]
  /-
    🎉 no goals
  -/


theorem ne_infty_iff_exists {x : OnePoint X} : x ≠ ∞ ↔ ∃ y : X, (y : OnePoint X) = x := by
  /-
    X : Type u_1
    x : OnePoint X
    ⊢ Iff (Ne x OnePoint.infty) (Exists fun y => Eq (↑y) x)
  -/
                                     /-
                                       🎉 no goals
                                     -/
  induction x using OnePoint.rec <;> simp
                                     /-
                                       🎉 no goals
                                     -/


instance canLift : CanLift (OnePoint X) X (↑) fun x => x ≠ ∞ :=
  WithTop.canLift


theorem not_mem_range_coe_iff {x : OnePoint X} : x ∉ range some ↔ x = ∞ := by
  /-
    X : Type u_1
    x : OnePoint X
    ⊢ Iff (Not (Membership.mem (Set.range OnePoint.some) x)) (Eq x OnePoint.infty)
  -/
  rw [← mem_compl_iff, compl_range_coe, mem_singleton_iff]
  /-
    🎉 no goals
  -/


theorem infty_not_mem_range_coe : ∞ ∉ range ((↑) : X → OnePoint X) :=
  not_mem_range_coe_iff.2 rfl


theorem infty_not_mem_image_coe {s : Set X} : ∞ ∉ ((↑) : X → OnePoint X) '' s :=
  not_mem_subset (image_subset_range _ _) infty_not_mem_range_coe


@[simp]
theorem coe_preimage_infty : ((↑) : X → OnePoint X) ⁻¹' {∞} = ∅ := by
  /-
    X : Type u_1
    ⊢ Eq (Set.preimage OnePoint.some (Singleton.singleton OnePoint.infty)) EmptyCo …
  -/
  ext
  /-
    case h
    X : Type u_1
    x✝ : X
    ⊢ Iff (Membership.mem (Set.preimage OnePoint.some (Singleton.singleton OnePoin …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Extend a map `f : X → Y` to a map `OnePoint X → OnePoint Y`
by sending infinity to infinity. -/
protected def map (f : X → Y) : OnePoint X → OnePoint Y :=
  Option.map f


@[simp] theorem map_infty (f : X → Y) : OnePoint.map f ∞ = ∞ := rfl

@[simp] theorem map_some (f : X → Y) (x : X) : (x : OnePoint X).map f = f x := rfl

@[simp] theorem map_id : OnePoint.map (id : X → X) = id := Option.map_id


theorem map_comp {Z : Type*} (f : Y → Z) (g : X → Y) :
    OnePoint.map (f ∘ g) = OnePoint.map f ∘ OnePoint.map g :=
  (Option.map_comp_map _ _).symm


instance : TopologicalSpace (OnePoint X) where
  IsOpen s := (∞ ∈ s → IsCompact (((↑) : X → OnePoint X) ⁻¹' s)ᶜ) ∧
    IsOpen (((↑) : X → OnePoint X) ⁻¹' s)
                    /-
                      X : Type u_1
                      Y : Type u_2
                      inst✝ : TopologicalSpace X
                      ⊢ (fun s => And (Membership.mem s OnePoint.infty → IsCompact (HasCompl.compl ( …
                    -/
  isOpen_univ := by simp
                    /-
                      🎉 no goals
                    -/
  isOpen_inter s t := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      s t : Set (OnePoint X)
      ⊢ (fun s => And (Membership.mem s OnePoint.infty → IsCompact (HasCompl.compl ( …
    -/
    rintro ⟨hms, hs⟩ ⟨hmt, ht⟩
    /-
      case intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      s t : Set (OnePoint X)
      hms : Membership.mem s OnePoint.infty → IsCompact (HasCompl.compl (Set.preimag …
      hs : IsOpen (Set.preimage OnePoint.some s)
      hmt : Membership.mem t OnePoint.infty → IsCompact (HasCompl.compl (Set.preimag …
      ht : IsOpen (Set.preimage OnePoint.some t)
      ⊢ And (Membership.mem (Inter.inter s t) OnePoint.infty → IsCompact (HasCompl.c …
    -/
    refine ⟨?_, hs.inter ht⟩
    /-
      case intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      s t : Set (OnePoint X)
      hms : Membership.mem s OnePoint.infty → IsCompact (HasCompl.compl (Set.preimag …
      hs : IsOpen (Set.preimage OnePoint.some s)
      hmt : Membership.mem t OnePoint.infty → IsCompact (HasCompl.compl (Set.preimag …
      ht : IsOpen (Set.preimage OnePoint.some t)
      ⊢ Membership.mem (Inter.inter s t) OnePoint.infty → IsCompact (HasCompl.compl  …
    -/
    rintro ⟨hms', hmt'⟩
    /-
      case intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      s t : Set (OnePoint X)
      hms : Membership.mem s OnePoint.infty → IsCompact (HasCompl.compl (Set.preimag …
      hs : IsOpen (Set.preimage OnePoint.some s)
      hmt : Membership.mem t OnePoint.infty → IsCompact (HasCompl.compl (Set.preimag …
      ht : IsOpen (Set.preimage OnePoint.some t)
      hms' : Membership.mem s OnePoint.infty
      hmt' : Membership.mem t OnePoint.infty
      ⊢ IsCompact (HasCompl.compl (Set.preimage OnePoint.some (Inter.inter s t)))
    -/
    simpa [compl_inter] using (hms hms').union (hmt hmt')
    /-
      🎉 no goals
    -/
  isOpen_sUnion S ho := by
    suffices IsOpen ((↑) ⁻¹' ⋃₀ S : Set X) by
      refine ⟨?_, this⟩
      rintro ⟨s, hsS : s ∈ S, hs : ∞ ∈ s⟩
      refine IsCompact.of_isClosed_subset ((ho s hsS).1 hs) this.isClosed_compl ?_
      exact compl_subset_compl.mpr (preimage_mono <| subset_sUnion_of_mem hsS)
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      S : Set (Set (OnePoint X))
      ho : ∀ (t : Set (OnePoint X)), Membership.mem S t → (fun s => And (Membership. …
      ⊢ IsOpen (Set.preimage OnePoint.some S.sUnion)
    -/
    rw [preimage_sUnion]
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      S : Set (Set (OnePoint X))
      ho : ∀ (t : Set (OnePoint X)), Membership.mem S t → (fun s => And (Membership. …
      ⊢ IsOpen (Set.iUnion fun t => Set.iUnion fun h => Set.preimage OnePoint.some t)
    -/
    exact isOpen_biUnion fun s hs => (ho s hs).2
    /-
      🎉 no goals
    -/


theorem isOpen_def :
    IsOpen s ↔ (∞ ∈ s → IsCompact ((↑) ⁻¹' s : Set X)ᶜ) ∧ IsOpen ((↑) ⁻¹' s : Set X) :=
  Iff.rfl


theorem isOpen_iff_of_mem' (h : ∞ ∈ s) :
    IsOpen s ↔ IsCompact ((↑) ⁻¹' s : Set X)ᶜ ∧ IsOpen ((↑) ⁻¹' s : Set X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    h : Membership.mem s OnePoint.infty
    ⊢ Iff (IsOpen s) (And (IsCompact (HasCompl.compl (Set.preimage OnePoint.some s …
  -/
  simp [isOpen_def, h]
  /-
    🎉 no goals
  -/


theorem isOpen_iff_of_mem (h : ∞ ∈ s) :
    IsOpen s ↔ IsClosed ((↑) ⁻¹' s : Set X)ᶜ ∧ IsCompact ((↑) ⁻¹' s : Set X)ᶜ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    h : Membership.mem s OnePoint.infty
    ⊢ Iff (IsOpen s) (And (IsClosed (HasCompl.compl (Set.preimage OnePoint.some s) …
  -/
  simp only [isOpen_iff_of_mem' h, isClosed_compl_iff, and_comm]
  /-
    🎉 no goals
  -/


theorem isOpen_iff_of_not_mem (h : ∞ ∉ s) : IsOpen s ↔ IsOpen ((↑) ⁻¹' s : Set X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    h : Not (Membership.mem s OnePoint.infty)
    ⊢ Iff (IsOpen s) (IsOpen (Set.preimage OnePoint.some s))
  -/
  simp [isOpen_def, h]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_of_mem (h : ∞ ∈ s) : IsClosed s ↔ IsClosed ((↑) ⁻¹' s : Set X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    h : Membership.mem s OnePoint.infty
    ⊢ Iff (IsClosed s) (IsClosed (Set.preimage OnePoint.some s))
  -/
  have : ∞ ∉ sᶜ := fun H => H h
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    h : Membership.mem s OnePoint.infty
    this : Not (Membership.mem (HasCompl.compl s) OnePoint.infty)
    ⊢ Iff (IsClosed s) (IsClosed (Set.preimage OnePoint.some s))
  -/
  rw [← isOpen_compl_iff, isOpen_iff_of_not_mem this, ← isOpen_compl_iff, preimage_compl]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_of_not_mem (h : ∞ ∉ s) :
    IsClosed s ↔ IsClosed ((↑) ⁻¹' s : Set X) ∧ IsCompact ((↑) ⁻¹' s : Set X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    h : Not (Membership.mem s OnePoint.infty)
    ⊢ Iff (IsClosed s) (And (IsClosed (Set.preimage OnePoint.some s)) (IsCompact ( …
  -/
  rw [← isOpen_compl_iff, isOpen_iff_of_mem (mem_compl h), ← preimage_compl, compl_compl]
  /-
    🎉 no goals
  -/


@[simp]
theorem isOpen_image_coe {s : Set X} : IsOpen ((↑) '' s : Set (OnePoint X)) ↔ IsOpen s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsOpen (Set.image OnePoint.some s)) (IsOpen s)
  -/
  rw [isOpen_iff_of_not_mem infty_not_mem_image_coe, preimage_image_eq _ coe_injective]
  /-
    🎉 no goals
  -/


theorem isOpen_compl_image_coe {s : Set X} :
    IsOpen ((↑) '' s : Set (OnePoint X))ᶜ ↔ IsClosed s ∧ IsCompact s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsOpen (HasCompl.compl (Set.image OnePoint.some s))) (And (IsClosed s)  …
  -/
  rw [isOpen_iff_of_mem, ← preimage_compl, compl_compl, preimage_image_eq _ coe_injective]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Membership.mem (HasCompl.compl (Set.image OnePoint.some s)) OnePoint.infty
  -/
  exact infty_not_mem_image_coe
  /-
    🎉 no goals
  -/


@[simp]
theorem isClosed_image_coe {s : Set X} :
    IsClosed ((↑) '' s : Set (OnePoint X)) ↔ IsClosed s ∧ IsCompact s := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsClosed (Set.image OnePoint.some s)) (And (IsClosed s) (IsCompact s))
  -/
  rw [← isOpen_compl_iff, isOpen_compl_image_coe]
  /-
    🎉 no goals
  -/


/-- An open set in `OnePoint X` constructed from a closed compact set in `X` -/
def opensOfCompl (s : Set X) (h₁ : IsClosed s) (h₂ : IsCompact s) :
    TopologicalSpace.Opens (OnePoint X) :=
  ⟨((↑) '' s)ᶜ, isOpen_compl_image_coe.2 ⟨h₁, h₂⟩⟩


theorem infty_mem_opensOfCompl {s : Set X} (h₁ : IsClosed s) (h₂ : IsCompact s) :
    ∞ ∈ opensOfCompl s h₁ h₂ :=
  mem_compl infty_not_mem_image_coe


@[continuity]
theorem continuous_coe : Continuous ((↑) : X → OnePoint X) :=
  continuous_def.mpr fun _s hs => hs.right


theorem isOpenMap_coe : IsOpenMap ((↑) : X → OnePoint X) := fun _ => isOpen_image_coe.2


theorem isOpenEmbedding_coe : IsOpenEmbedding ((↑) : X → OnePoint X) :=
  .of_continuous_injective_isOpenMap continuous_coe coe_injective isOpenMap_coe


@[deprecated (since := "2024-10-18")]
alias openEmbedding_coe := isOpenEmbedding_coe


theorem isOpen_range_coe : IsOpen (range ((↑) : X → OnePoint X)) :=
  isOpenEmbedding_coe.isOpen_range


theorem isClosed_infty : IsClosed ({∞} : Set (OnePoint X)) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ IsClosed (Singleton.singleton OnePoint.infty)
  -/
  rw [← compl_range_coe, isClosed_compl_iff]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ IsOpen (Set.range OnePoint.some)
  -/
  exact isOpen_range_coe
  /-
    🎉 no goals
  -/


theorem nhds_coe_eq (x : X) : 𝓝 ↑x = map ((↑) : X → OnePoint X) (𝓝 x) :=
  (isOpenEmbedding_coe.map_nhds_eq x).symm


theorem nhdsWithin_coe_image (s : Set X) (x : X) :
    𝓝[(↑) '' s] (x : OnePoint X) = map (↑) (𝓝[s] x) :=
  (isOpenEmbedding_coe.isEmbedding.map_nhdsWithin_eq _ _).symm


theorem nhdsWithin_coe (s : Set (OnePoint X)) (x : X) : 𝓝[s] ↑x = map (↑) (𝓝[(↑) ⁻¹' s] x) :=
  (isOpenEmbedding_coe.map_nhdsWithin_preimage_eq _ _).symm


theorem comap_coe_nhds (x : X) : comap ((↑) : X → OnePoint X) (𝓝 x) = 𝓝 x :=
  (isOpenEmbedding_coe.isInducing.nhds_eq_comap x).symm


/-- If `x` is not an isolated point of `X`, then `x : OnePoint X` is not an isolated point
of `OnePoint X`. -/
instance nhdsWithin_compl_coe_neBot (x : X) [h : NeBot (𝓝[≠] x)] :
    NeBot (𝓝[≠] (x : OnePoint X)) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    s : Set (OnePoint X)
    x : X
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
    ⊢ (nhdsWithin (↑x) (HasCompl.compl (Singleton.singleton ↑x))).NeBot
  -/
  simpa [nhdsWithin_coe, preimage, coe_eq_coe] using h.map some
  /-
    🎉 no goals
  -/


theorem nhdsWithin_compl_infty_eq : 𝓝[≠] (∞ : OnePoint X) = map (↑) (coclosedCompact X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq (nhdsWithin OnePoint.infty (HasCompl.compl (Singleton.singleton OnePoint. …
  -/
  refine (nhdsWithin_basis_open ∞ _).ext (hasBasis_coclosedCompact.map _) ?_ ?_
    /-
      case refine_1
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ ∀ (i : Set (OnePoint X)), And (Membership.mem i OnePoint.infty) (IsOpen i) → …
    -/
  · rintro s ⟨hs, hso⟩
    /-
      case refine_1.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set (OnePoint X)
      hs : Membership.mem s OnePoint.infty
      hso : IsOpen s
      ⊢ Exists fun i' => And (And (IsClosed i') (IsCompact i')) (HasSubset.Subset (S …
    -/
    refine ⟨_, (isOpen_iff_of_mem hs).mp hso, ?_⟩
    /-
      case refine_1.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set (OnePoint X)
      hs : Membership.mem s OnePoint.infty
      hso : IsOpen s
      ⊢ HasSubset.Subset (Set.image OnePoint.some (HasCompl.compl (HasCompl.compl (S …
    -/
    simp [Subset.rfl]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ ∀ (i' : Set X), And (IsClosed i') (IsCompact i') → Exists fun i => And (And  …
    -/
  · rintro s ⟨h₁, h₂⟩
    /-
      case refine_2.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      h₁ : IsClosed s
      h₂ : IsCompact s
      ⊢ Exists fun i => And (And (Membership.mem i OnePoint.infty) (IsOpen i)) (HasS …
    -/
    refine ⟨_, ⟨mem_compl infty_not_mem_image_coe, isOpen_compl_image_coe.2 ⟨h₁, h₂⟩⟩, ?_⟩
    /-
      case refine_2.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      h₁ : IsClosed s
      h₂ : IsCompact s
      ⊢ HasSubset.Subset (Inter.inter (HasCompl.compl (Set.image OnePoint.some s)) ( …
    -/
    simp [compl_image_coe, ← diff_eq, subset_preimage_image]
    /-
      🎉 no goals
    -/


/-- If `X` is a non-compact space, then `∞` is not an isolated point of `OnePoint X`. -/
instance nhdsWithin_compl_infty_neBot [NoncompactSpace X] : NeBot (𝓝[≠] (∞ : OnePoint X)) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝ : NoncompactSpace X
    ⊢ (nhdsWithin OnePoint.infty (HasCompl.compl (Singleton.singleton OnePoint.inf …
  -/
  rw [nhdsWithin_compl_infty_eq]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝ : NoncompactSpace X
    ⊢ (Filter.map OnePoint.some (Filter.coclosedCompact X)).NeBot
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (priority := 900) nhdsWithin_compl_neBot [∀ x : X, NeBot (𝓝[≠] x)] [NoncompactSpace X]
    (x : OnePoint X) : NeBot (𝓝[≠] x) :=
  OnePoint.rec OnePoint.nhdsWithin_compl_infty_neBot
    (fun y => OnePoint.nhdsWithin_compl_coe_neBot y) x


theorem nhds_infty_eq : 𝓝 (∞ : OnePoint X) = map (↑) (coclosedCompact X) ⊔ pure ∞ := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq (nhds OnePoint.infty) (Max.max (Filter.map OnePoint.some (Filter.coclosed …
  -/
  rw [← nhdsWithin_compl_infty_eq, nhdsWithin_compl_singleton_sup_pure]
  /-
    🎉 no goals
  -/


theorem tendsto_coe_infty : Tendsto (↑) (coclosedCompact X) (𝓝 (∞ : OnePoint X)) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Filter.Tendsto OnePoint.some (Filter.coclosedCompact X) (nhds OnePoint.infty)
  -/
  rw [nhds_infty_eq]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Filter.Tendsto OnePoint.some (Filter.coclosedCompact X) (Max.max (Filter.map …
  -/
  exact Filter.Tendsto.mono_right tendsto_map le_sup_left
  /-
    🎉 no goals
  -/


theorem hasBasis_nhds_infty :
    (𝓝 (∞ : OnePoint X)).HasBasis (fun s : Set X => IsClosed s ∧ IsCompact s) fun s =>
      (↑) '' sᶜ ∪ {∞} := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ (nhds OnePoint.infty).HasBasis (fun s => And (IsClosed s) (IsCompact s)) fun …
  -/
  rw [nhds_infty_eq]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ (Max.max (Filter.map OnePoint.some (Filter.coclosedCompact X)) (Pure.pure On …
  -/
  exact (hasBasis_coclosedCompact.map _).sup_pure _
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_coe_nhds_infty : comap ((↑) : X → OnePoint X) (𝓝 ∞) = coclosedCompact X := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq (Filter.comap OnePoint.some (nhds OnePoint.infty)) (Filter.coclosedCompac …
  -/
  simp [nhds_infty_eq, comap_sup, comap_map coe_injective]
  /-
    🎉 no goals
  -/


theorem le_nhds_infty {f : Filter (OnePoint X)} :
    f ≤ 𝓝 ∞ ↔ ∀ s : Set X, IsClosed s → IsCompact s → (↑) '' sᶜ ∪ {∞} ∈ f := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    f : Filter (OnePoint X)
    ⊢ Iff (LE.le f (nhds OnePoint.infty)) (∀ (s : Set X), IsClosed s → IsCompact s …
  -/
  simp only [hasBasis_nhds_infty.ge_iff, and_imp]
  /-
    🎉 no goals
  -/


theorem ultrafilter_le_nhds_infty {f : Ultrafilter (OnePoint X)} :
    (f : Filter (OnePoint X)) ≤ 𝓝 ∞ ↔ ∀ s : Set X, IsClosed s → IsCompact s → (↑) '' s ∉ f := by
  simp only [le_nhds_infty, ← compl_image_coe, Ultrafilter.mem_coe,
    Ultrafilter.compl_mem_iff_not_mem]


theorem tendsto_nhds_infty' {α : Type*} {f : OnePoint X → α} {l : Filter α} :
    Tendsto f (𝓝 ∞) l ↔ Tendsto f (pure ∞) l ∧ Tendsto (f ∘ (↑)) (coclosedCompact X) l := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    α : Type u_3
    f : OnePoint X → α
    l : Filter α
    ⊢ Iff (Filter.Tendsto f (nhds OnePoint.infty) l) (And (Filter.Tendsto f (Pure. …
  -/
  simp [nhds_infty_eq, and_comm]
  /-
    🎉 no goals
  -/


theorem tendsto_nhds_infty {α : Type*} {f : OnePoint X → α} {l : Filter α} :
    Tendsto f (𝓝 ∞) l ↔
      ∀ s ∈ l, f ∞ ∈ s ∧ ∃ t : Set X, IsClosed t ∧ IsCompact t ∧ MapsTo (f ∘ (↑)) tᶜ s :=
  tendsto_nhds_infty'.trans <| by
    simp only [tendsto_pure_left, hasBasis_coclosedCompact.tendsto_left_iff, forall_and,
      and_assoc, exists_prop]


theorem continuousAt_infty' {Y : Type*} [TopologicalSpace Y] {f : OnePoint X → Y} :
    ContinuousAt f ∞ ↔ Tendsto (f ∘ (↑)) (coclosedCompact X) (𝓝 (f ∞)) :=
  tendsto_nhds_infty'.trans <| and_iff_right (tendsto_pure_nhds _ _)


theorem continuousAt_infty {Y : Type*} [TopologicalSpace Y] {f : OnePoint X → Y} :
    ContinuousAt f ∞ ↔
      ∀ s ∈ 𝓝 (f ∞), ∃ t : Set X, IsClosed t ∧ IsCompact t ∧ MapsTo (f ∘ (↑)) tᶜ s :=
                                  /-
                                    X : Type u_1
                                    inst✝¹ : TopologicalSpace X
                                    Y : Type u_3
                                    inst✝ : TopologicalSpace Y
                                    f : OnePoint X → Y
                                    ⊢ Iff (Filter.Tendsto (Function.comp f OnePoint.some) (Filter.coclosedCompact  …
                                  -/
  continuousAt_infty'.trans <| by simp only [hasBasis_coclosedCompact.tendsto_left_iff, and_assoc]
                                  /-
                                    🎉 no goals
                                  -/


theorem continuousAt_coe {Y : Type*} [TopologicalSpace Y] {f : OnePoint X → Y} {x : X} :
    ContinuousAt f x ↔ ContinuousAt (f ∘ (↑)) x := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    Y : Type u_3
    inst✝ : TopologicalSpace Y
    f : OnePoint X → Y
    x : X
    ⊢ Iff (ContinuousAt f ↑x) (ContinuousAt (Function.comp f OnePoint.some) x)
  -/
  rw [ContinuousAt, nhds_coe_eq, tendsto_map'_iff, ContinuousAt]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma continuous_iff {Y : Type*} [TopologicalSpace Y] (f : OnePoint X → Y) : Continuous f ↔
    Tendsto (fun x : X ↦ f x) (coclosedCompact X) (𝓝 (f ∞)) ∧ Continuous (fun x : X ↦ f x) := by
  simp only [continuous_iff_continuousAt, OnePoint.forall, continuousAt_coe, continuousAt_infty',
    Function.comp_def]


/--
A constructor for continuous maps out of a one point compactification, given a continuous map from
the underlying space and a limit value at infinity.
-/
def continuousMapMk {Y : Type*} [TopologicalSpace Y] (f : C(X, Y)) (y : Y)
    (h : Tendsto f (coclosedCompact X) (𝓝 y)) : C(OnePoint X, Y) where
  toFun x := x.elim y f
  continuous_toFun := by
    /-
      X : Type u_1
      Y✝ : Type u_2
      inst✝¹ : TopologicalSpace X
      s : Set (OnePoint X)
      Y : Type u_3
      inst✝ : TopologicalSpace Y
      f : ContinuousMap X Y
      y : Y
      h : Filter.Tendsto (⇑f) (Filter.coclosedCompact X) (nhds y)
      ⊢ Continuous fun x => x.elim y ⇑f
    -/
    rw [continuous_iff]
    /-
      X : Type u_1
      Y✝ : Type u_2
      inst✝¹ : TopologicalSpace X
      s : Set (OnePoint X)
      Y : Type u_3
      inst✝ : TopologicalSpace Y
      f : ContinuousMap X Y
      y : Y
      h : Filter.Tendsto (⇑f) (Filter.coclosedCompact X) (nhds y)
      ⊢ And (Filter.Tendsto (fun x => (↑x).elim y ⇑f) (Filter.coclosedCompact X) (nh …
    -/
    refine ⟨h, f.continuous⟩
    /-
      🎉 no goals
    -/


lemma continuous_iff_from_discrete {Y : Type*} [TopologicalSpace Y]
    [DiscreteTopology X] (f : OnePoint X → Y) :
    Continuous f ↔ Tendsto (fun x : X ↦ f x) cofinite (𝓝 (f ∞)) := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    Y : Type u_3
    inst✝¹ : TopologicalSpace Y
    inst✝ : DiscreteTopology X
    f : OnePoint X → Y
    ⊢ Iff (Continuous f) (Filter.Tendsto (fun x => f ↑x) Filter.cofinite (nhds (f  …
  -/
  simp [continuous_iff, cocompact_eq_cofinite, continuous_of_discreteTopology]
  /-
    🎉 no goals
  -/


/--
A constructor for continuous maps out of a one point compactification of a discrete space, given a
map from the underlying space and a limit value at infinity.
-/
def continuousMapMkDiscrete {Y : Type*} [TopologicalSpace Y]
    [DiscreteTopology X] (f : X → Y) (y : Y) (h : Tendsto f cofinite (𝓝 y)) :
    C(OnePoint X, Y) :=
                                                            /-
                                                              X : Type u_1
                                                              Y✝ : Type u_2
                                                              inst✝² : TopologicalSpace X
                                                              s : Set (OnePoint X)
                                                              Y : Type u_3
                                                              inst✝¹ : TopologicalSpace Y
                                                              inst✝ : DiscreteTopology X
                                                              f : X → Y
                                                              y : Y
                                                              h : Filter.Tendsto f Filter.cofinite (nhds y)
                                                              ⊢ Filter.Tendsto (⇑{ toFun := f, continuous_toFun := ⋯ }) (Filter.coclosedComp …
                                                            -/
  continuousMapMk ⟨f, continuous_of_discreteTopology⟩ y (by simpa [cocompact_eq_cofinite])
                                                            /-
                                                              🎉 no goals
                                                            -/


variable (X) in
/--
Continuous maps out of the one point compactification of an infinite discrete space to a Hausdorff
space correspond bijectively to "convergent" maps out of the discrete space.
-/
noncomputable def continuousMapDiscreteEquiv (Y : Type*) [DiscreteTopology X] [TopologicalSpace Y]
    [T2Space Y] [Infinite X] :
    C(OnePoint X, Y) ≃ { f : X → Y // ∃ L, Tendsto (fun x : X ↦ f x) cofinite (𝓝 L) } where
  toFun f := ⟨(f ·), ⟨f ∞, continuous_iff_from_discrete _ |>.mp (map_continuous f)⟩⟩
  invFun f :=
    { toFun := fun x => match x with
        | ∞ => Classical.choose f.2
        | some x => f.1 x
      continuous_toFun := continuous_iff_from_discrete _ |>.mpr <| Classical.choose_spec f.2 }
  left_inv f := by
    /-
      X : Type u_1
      Y✝ : Type u_2
      inst✝⁴ : TopologicalSpace X
      s : Set (OnePoint X)
      Y : Type u_3
      inst✝³ : DiscreteTopology X
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : Infinite X
      f : ContinuousMap (OnePoint X) Y
      ⊢ Eq ((fun f => { toFun := fun x => instReprOnePoint.match_1 (fun x => Y) x (f …
    -/
    ext x
    /-
      case h
      X : Type u_1
      Y✝ : Type u_2
      inst✝⁴ : TopologicalSpace X
      s : Set (OnePoint X)
      Y : Type u_3
      inst✝³ : DiscreteTopology X
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : Infinite X
      f : ContinuousMap (OnePoint X) Y
      x : OnePoint X
      ⊢ Eq (((fun f => { toFun := fun x => instReprOnePoint.match_1 (fun x => Y) x ( …
    -/
    refine OnePoint.rec ?_ ?_ x
      /-
        case h.refine_1
        X : Type u_1
        Y✝ : Type u_2
        inst✝⁴ : TopologicalSpace X
        s : Set (OnePoint X)
        Y : Type u_3
        inst✝³ : DiscreteTopology X
        inst✝² : TopologicalSpace Y
        inst✝¹ : T2Space Y
        inst✝ : Infinite X
        f : ContinuousMap (OnePoint X) Y
        x : OnePoint X
        ⊢ Eq (((fun f => { toFun := fun x => instReprOnePoint.match_1 (fun x => Y) x ( …
      -/
    · refine tendsto_nhds_unique ?_ (continuous_iff_from_discrete _ |>.mp <| map_continuous f)
      let f' : { f : X → Y // ∃ L, Tendsto (fun x : X ↦ f x) cofinite (𝓝 L) } :=
        ⟨fun x ↦ f x, ⟨f ∞, continuous_iff_from_discrete f |>.mp <| map_continuous f⟩⟩
      /-
        case h.refine_1
        X : Type u_1
        Y✝ : Type u_2
        inst✝⁴ : TopologicalSpace X
        s : Set (OnePoint X)
        Y : Type u_3
        inst✝³ : DiscreteTopology X
        inst✝² : TopologicalSpace Y
        inst✝¹ : T2Space Y
        inst✝ : Infinite X
        f : ContinuousMap (OnePoint X) Y
        x : OnePoint X
        f' : Subtype fun f => Exists fun L => Filter.Tendsto (fun x => f x) Filter.cof …
        ⊢ Filter.Tendsto (fun x => f ↑x) Filter.cofinite (nhds (((fun f => { toFun :=  …
      -/
      exact Classical.choose_spec f'.property
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        X : Type u_1
        Y✝ : Type u_2
        inst✝⁴ : TopologicalSpace X
        s : Set (OnePoint X)
        Y : Type u_3
        inst✝³ : DiscreteTopology X
        inst✝² : TopologicalSpace Y
        inst✝¹ : T2Space Y
        inst✝ : Infinite X
        f : ContinuousMap (OnePoint X) Y
        x : OnePoint X
        ⊢ ∀ (x : X), Eq (((fun f => { toFun := fun x => instReprOnePoint.match_1 (fun  …
      -/
    · simp
      /-
        🎉 no goals
      -/
  right_inv _ := rfl


lemma continuous_iff_from_nat {Y : Type*} [TopologicalSpace Y] (f : OnePoint ℕ → Y) :
    Continuous f ↔ Tendsto (fun x : ℕ ↦ f x) atTop (𝓝 (f ∞)) := by
  /-
    Y : Type u_3
    inst✝ : TopologicalSpace Y
    f : OnePoint Nat → Y
    ⊢ Iff (Continuous f) (Filter.Tendsto (fun x => f ↑x) Filter.atTop (nhds (f One …
  -/
  rw [continuous_iff_from_discrete, Nat.cofinite_eq_atTop]
  /-
    🎉 no goals
  -/


/--
A constructor for continuous maps out of the one point compactification of `ℕ`, given a
sequence and a limit value at infinity.
-/
def continuousMapMkNat {Y : Type*} [TopologicalSpace Y]
    (f : ℕ → Y) (y : Y) (h : Tendsto f atTop (𝓝 y)) :
    C(OnePoint ℕ, Y) :=
                                  /-
                                    X : Type u_1
                                    Y✝ : Type u_2
                                    inst✝¹ : TopologicalSpace X
                                    s : Set (OnePoint X)
                                    Y : Type u_3
                                    inst✝ : TopologicalSpace Y
                                    f : Nat → Y
                                    y : Y
                                    h : Filter.Tendsto f Filter.atTop (nhds y)
                                    ⊢ Filter.Tendsto f Filter.cofinite (nhds y)
                                  -/
  continuousMapMkDiscrete f y (by rwa [Nat.cofinite_eq_atTop])
                                  /-
                                    🎉 no goals
                                  -/


/--
Continuous maps out of the one point compactification of `ℕ` to a Hausdorff space `Y` correspond
bijectively to convergent sequences in `Y`.
-/
noncomputable def continuousMapNatEquiv (Y : Type*) [TopologicalSpace Y] [T2Space Y] :
    C(OnePoint ℕ, Y) ≃ { f : ℕ → Y // ∃ L, Tendsto (f ·) atTop (𝓝 L) } := by
  refine (continuousMapDiscreteEquiv ℕ Y).trans {
    toFun := fun ⟨f, hf⟩ ↦ ⟨f, by rwa [← Nat.cofinite_eq_atTop]⟩
    invFun := fun ⟨f, hf⟩ ↦ ⟨f, by rwa [Nat.cofinite_eq_atTop]⟩
    left_inv := fun _ ↦ rfl
    right_inv := fun _ ↦ rfl }


/-- If `X` is not a compact space, then the natural embedding `X → OnePoint X` has dense range.
-/
theorem denseRange_coe [NoncompactSpace X] : DenseRange ((↑) : X → OnePoint X) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : NoncompactSpace X
    ⊢ DenseRange OnePoint.some
  -/
  rw [DenseRange, ← compl_infty]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : NoncompactSpace X
    ⊢ Dense (HasCompl.compl (Singleton.singleton OnePoint.infty))
  -/
  exact dense_compl_singleton _
  /-
    🎉 no goals
  -/


theorem isDenseEmbedding_coe [NoncompactSpace X] : IsDenseEmbedding ((↑) : X → OnePoint X) :=
  { isOpenEmbedding_coe with dense := denseRange_coe }


@[deprecated (since := "2024-09-30")]
alias denseEmbedding_coe := isDenseEmbedding_coe


@[simp, norm_cast]
theorem specializes_coe {x y : X} : (x : OnePoint X) ⤳ y ↔ x ⤳ y :=
  isOpenEmbedding_coe.isInducing.specializes_iff


@[simp, norm_cast]
theorem inseparable_coe {x y : X} : Inseparable (x : OnePoint X) y ↔ Inseparable x y :=
  isOpenEmbedding_coe.isInducing.inseparable_iff


theorem not_specializes_infty_coe {x : X} : ¬Specializes ∞ (x : OnePoint X) :=
  isClosed_infty.not_specializes rfl (coe_ne_infty x)


theorem not_inseparable_infty_coe {x : X} : ¬Inseparable ∞ (x : OnePoint X) := fun h =>
  not_specializes_infty_coe h.specializes


theorem not_inseparable_coe_infty {x : X} : ¬Inseparable (x : OnePoint X) ∞ := fun h =>
  not_specializes_infty_coe h.specializes'


theorem inseparable_iff {x y : OnePoint X} :
    Inseparable x y ↔ x = ∞ ∧ y = ∞ ∨ ∃ x' : X, x = x' ∧ ∃ y' : X, y = y' ∧ Inseparable x' y' := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    x y : OnePoint X
    ⊢ Iff (Inseparable x y) (Or (And (Eq x OnePoint.infty) (Eq y OnePoint.infty))  …
  -/
  induction x using OnePoint.rec <;> induction y using OnePoint.rec <;>
    /-
      case infty.infty
      X : Type u_1
      inst✝ : TopologicalSpace X
      ⊢ Iff (Inseparable OnePoint.infty OnePoint.infty) (Or (And (Eq OnePoint.infty  …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [not_inseparable_infty_coe, not_inseparable_coe_infty, coe_eq_coe, Inseparable.refl]
    /-
      🎉 no goals
    -/


theorem continuous_map_iff [TopologicalSpace Y] {f : X → Y} :
    Continuous (OnePoint.map f) ↔
      Continuous f ∧ Tendsto f (coclosedCompact X) (coclosedCompact Y) := by
  simp_rw [continuous_iff, map_some, ← comap_coe_nhds_infty, tendsto_comap_iff, map_infty,
    isOpenEmbedding_coe.isInducing.continuous_iff (Y := Y)]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (And (Filter.Tendsto (fun x => ↑(f x)) (Filter.comap OnePoint.some (nhds …
  -/
  exact and_comm
  /-
    🎉 no goals
  -/


theorem continuous_map [TopologicalSpace Y] {f : X → Y} (hc : Continuous f)
    (h : Tendsto f (coclosedCompact X) (coclosedCompact Y)) :
    Continuous (OnePoint.map f) :=
  continuous_map_iff.mpr ⟨hc, h⟩


/-- For any topological space `X`, its one point compactification is a compact space. -/
instance : CompactSpace (OnePoint X) where
  isCompact_univ := by
    have : Tendsto ((↑) : X → OnePoint X) (cocompact X) (𝓝 ∞) := by
      rw [nhds_infty_eq]
      exact (tendsto_map.mono_left cocompact_le_coclosedCompact).mono_right le_sup_left
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      s : Set (OnePoint X)
      this : Filter.Tendsto OnePoint.some (Filter.cocompact X) (nhds OnePoint.infty)
      ⊢ IsCompact Set.univ
    -/
    rw [← insert_none_range_some X]
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      s : Set (OnePoint X)
      this : Filter.Tendsto OnePoint.some (Filter.cocompact X) (nhds OnePoint.infty)
      ⊢ IsCompact (Insert.insert Option.none (Set.range Option.some))
    -/
    exact this.isCompact_insert_range_of_cocompact continuous_coe
    /-
      🎉 no goals
    -/


/-- The one point compactification of a `T0Space` space is a `T0Space`. -/
instance [T0Space X] : T0Space (OnePoint X) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝ : T0Space X
    ⊢ T0Space (OnePoint X)
  -/
  refine ⟨fun x y hxy => ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝ : T0Space X
    x y : OnePoint X
    hxy : Inseparable x y
    ⊢ Eq x y
  -/
  rcases inseparable_iff.1 hxy with (⟨rfl, rfl⟩ | ⟨x, rfl, y, rfl, h⟩)
  /-
    case inl.intro
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝ : T0Space X
    hxy : Inseparable OnePoint.infty OnePoint.infty
    ⊢ Eq OnePoint.infty OnePoint.infty
  -/
  exacts [rfl, congr_arg some h.eq]
  /-
    🎉 no goals
  -/


/-- The one point compactification of a `T1Space` space is a `T1Space`. -/
instance [T1Space X] : T1Space (OnePoint X) where
  t1 z := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝ : T1Space X
      z : OnePoint X
      ⊢ IsClosed (Singleton.singleton z)
    -/
    induction z using OnePoint.rec
      /-
        case infty
        X : Type u_1
        Y : Type u_2
        inst✝¹ : TopologicalSpace X
        s : Set (OnePoint X)
        inst✝ : T1Space X
        ⊢ IsClosed (Singleton.singleton OnePoint.infty)
      -/
    · exact isClosed_infty
      /-
        🎉 no goals
      -/
      /-
        case coe
        X : Type u_1
        Y : Type u_2
        inst✝¹ : TopologicalSpace X
        s : Set (OnePoint X)
        inst✝ : T1Space X
        x✝ : X
        ⊢ IsClosed (Singleton.singleton ↑x✝)
      -/
    · rw [← image_singleton, isClosed_image_coe]
      /-
        case coe
        X : Type u_1
        Y : Type u_2
        inst✝¹ : TopologicalSpace X
        s : Set (OnePoint X)
        inst✝ : T1Space X
        x✝ : X
        ⊢ And (IsClosed (Singleton.singleton x✝)) (IsCompact (Singleton.singleton x✝))
      -/
      exact ⟨isClosed_singleton, isCompact_singleton⟩
      /-
        🎉 no goals
      -/


/-- The one point compactification of a locally compact R₁ space is a normal topological space. -/
instance [LocallyCompactSpace X] [R1Space X] : NormalSpace (OnePoint X) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝¹ : LocallyCompactSpace X
    inst✝ : R1Space X
    ⊢ NormalSpace (OnePoint X)
  -/
  suffices R1Space (OnePoint X) by infer_instance
  have key : ∀ z : X, Disjoint (𝓝 (some z)) (𝓝 ∞) := fun z ↦ by
    rw [nhds_infty_eq, disjoint_sup_right, nhds_coe_eq, coclosedCompact_eq_cocompact,
      disjoint_map coe_injective, ← principal_singleton, disjoint_principal_right, compl_infty]
    exact ⟨disjoint_nhds_cocompact z, range_mem_map⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝¹ : LocallyCompactSpace X
    inst✝ : R1Space X
    key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
    ⊢ R1Space (OnePoint X)
  -/
  refine ⟨fun x y ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    s : Set (OnePoint X)
    inst✝¹ : LocallyCompactSpace X
    inst✝ : R1Space X
    key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
    x y : OnePoint X
    ⊢ Or (Specializes x y) (Disjoint (nhds x) (nhds y))
  -/
  induction x using OnePoint.rec <;> induction y using OnePoint.rec
    /-
      case infty.infty
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝¹ : LocallyCompactSpace X
      inst✝ : R1Space X
      key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
      ⊢ Or (Specializes OnePoint.infty OnePoint.infty) (Disjoint (nhds OnePoint.inft …
    -/
  · exact .inl le_rfl
    /-
      🎉 no goals
    -/
    /-
      case infty.coe
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝¹ : LocallyCompactSpace X
      inst✝ : R1Space X
      key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
      x✝ : X
      ⊢ Or (Specializes OnePoint.infty ↑x✝) (Disjoint (nhds OnePoint.infty) (nhds ↑x …
    -/
  · exact .inr (key _).symm
    /-
      🎉 no goals
    -/
    /-
      case coe.infty
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝¹ : LocallyCompactSpace X
      inst✝ : R1Space X
      key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
      x✝ : X
      ⊢ Or (Specializes (↑x✝) OnePoint.infty) (Disjoint (nhds ↑x✝) (nhds OnePoint.in …
    -/
  · exact .inr (key _)
    /-
      🎉 no goals
    -/
    /-
      case coe.coe
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝¹ : LocallyCompactSpace X
      inst✝ : R1Space X
      key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
      x✝¹ x✝ : X
      ⊢ Or (Specializes ↑x✝¹ ↑x✝) (Disjoint (nhds ↑x✝¹) (nhds ↑x✝))
    -/
  · rw [nhds_coe_eq, nhds_coe_eq, disjoint_map coe_injective, specializes_coe]
    /-
      case coe.coe
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝¹ : LocallyCompactSpace X
      inst✝ : R1Space X
      key : ∀ (z : X), Disjoint (nhds ↑z) (nhds OnePoint.infty)
      x✝¹ x✝ : X
      ⊢ Or (Specializes x✝¹ x✝) (Disjoint (nhds x✝¹) (nhds x✝))
    -/
    apply specializes_or_disjoint_nhds
    /-
      🎉 no goals
    -/


/-- If `X` is not a compact space, then `OnePoint X` is a connected space. -/
instance [PreconnectedSpace X] [NoncompactSpace X] : ConnectedSpace (OnePoint X) where
  toPreconnectedSpace := isDenseEmbedding_coe.isDenseInducing.preconnectedSpace
  toNonempty := inferInstance


/-- If `X` is an infinite type with discrete topology (e.g., `ℕ`), then the identity map from
`CofiniteTopology (OnePoint X)` to `OnePoint X` is not continuous. -/
theorem not_continuous_cofiniteTopology_of_symm [Infinite X] [DiscreteTopology X] :
    ¬Continuous (@CofiniteTopology.of (OnePoint X)).symm := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : Infinite X
    inst✝ : DiscreteTopology X
    ⊢ Not (Continuous ⇑CofiniteTopology.of.symm)
  -/
  inhabit X
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : Infinite X
    inst✝ : DiscreteTopology X
    inhabited_h : Inhabited X
    ⊢ Not (Continuous ⇑CofiniteTopology.of.symm)
  -/
  simp only [continuous_iff_continuousAt, ContinuousAt, not_forall]
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : Infinite X
    inst✝ : DiscreteTopology X
    inhabited_h : Inhabited X
    ⊢ Exists fun x => Not (Filter.Tendsto (⇑CofiniteTopology.of.symm) (nhds x) (nh …
  -/
  use CofiniteTopology.of ↑(default : X)
  simpa [nhds_coe_eq, nhds_discrete, CofiniteTopology.nhds_eq] using
    (finite_singleton ((default : X) : OnePoint X)).infinite_compl


instance (X : Type*) [TopologicalSpace X] [DiscreteTopology X] :
    TotallySeparatedSpace (OnePoint X) where
  isTotallySeparated_univ x _ y _ hxy := by
    cases x with
    | infty =>
      refine ⟨{y}ᶜ, {y}, isOpen_compl_singleton, ?_, hxy, rfl, (compl_union_self _).symm.subset,
        disjoint_compl_left⟩
      rw [OnePoint.isOpen_iff_of_not_mem]
      exacts [isOpen_discrete _, hxy]
    | coe val =>
      refine ⟨{some val}, {some val}ᶜ, ?_, isOpen_compl_singleton, rfl, hxy.symm, by simp,
        disjoint_compl_right⟩
      rw [OnePoint.isOpen_iff_of_not_mem]
      exacts [isOpen_discrete _, (Option.some_ne_none val).symm]


open scoped Classical in
/-- If `f` embeds `X` into a compact Hausdorff space `Y`, and has exactly one point outside its
range, then `(Y, f)` is the one-point compactification of `X`. -/
noncomputable def equivOfIsEmbeddingOfRangeEq :
    OnePoint X ≃ₜ Y :=
  have _i := hf.t2Space
  have : Tendsto f (coclosedCompact X) (𝓝 y) := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : CompactSpace Y
      y : Y
      f : X → Y
      hf : Topology.IsEmbedding f
      hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
      _i : T2Space X
      ⊢ Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
    -/
    rw [coclosedCompact_eq_cocompact, hasBasis_cocompact.tendsto_left_iff]
    /-
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : CompactSpace Y
      y : Y
      f : X → Y
      hf : Topology.IsEmbedding f
      hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
      _i : T2Space X
      ⊢ ∀ (t : Set Y), Membership.mem (nhds y) t → Exists fun i => And (IsCompact i) …
    -/
    intro N hN
    /-
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : CompactSpace Y
      y : Y
      f : X → Y
      hf : Topology.IsEmbedding f
      hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
      _i : T2Space X
      N : Set Y
      hN : Membership.mem (nhds y) N
      ⊢ Exists fun i => And (IsCompact i) (Set.MapsTo f (HasCompl.compl i) N)
    -/
    obtain ⟨U, hU₁, hU₂, hU₃⟩ := mem_nhds_iff.mp hN
    /-
      case intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : CompactSpace Y
      y : Y
      f : X → Y
      hf : Topology.IsEmbedding f
      hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
      _i : T2Space X
      N : Set Y
      hN : Membership.mem (nhds y) N
      U : Set Y
      hU₁ : HasSubset.Subset U N
      hU₂ : IsOpen U
      hU₃ : Membership.mem U y
      ⊢ Exists fun i => And (IsCompact i) (Set.MapsTo f (HasCompl.compl i) N)
    -/
    refine ⟨f⁻¹' Uᶜ, ?_, by simpa using (mapsTo_preimage f U).mono_right hU₁⟩
    /-
      case intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : CompactSpace Y
      y : Y
      f : X → Y
      hf : Topology.IsEmbedding f
      hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
      _i : T2Space X
      N : Set Y
      hN : Membership.mem (nhds y) N
      U : Set Y
      hU₁ : HasSubset.Subset U N
      hU₂ : IsOpen U
      hU₃ : Membership.mem U y
      ⊢ IsCompact (Set.preimage f (HasCompl.compl U))
    -/
    rw [hf.isCompact_iff, image_preimage_eq_iff.mpr (by simpa [hy])]
    /-
      case intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      s : Set (OnePoint X)
      inst✝² : TopologicalSpace Y
      inst✝¹ : T2Space Y
      inst✝ : CompactSpace Y
      y : Y
      f : X → Y
      hf : Topology.IsEmbedding f
      hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
      _i : T2Space X
      N : Set Y
      hN : Membership.mem (nhds y) N
      U : Set Y
      hU₁ : HasSubset.Subset U N
      hU₂ : IsOpen U
      hU₃ : Membership.mem U y
      ⊢ IsCompact (HasCompl.compl U)
    -/
    exact (isClosed_compl_iff.mpr hU₂).isCompact
    /-
      🎉 no goals
    -/
  let e : OnePoint X ≃ Y :=
    { toFun := fun p ↦ p.elim y f
                                                                             /-
                                                                               X : Type u_1
                                                                               Y : Type u_2
                                                                               inst✝³ : TopologicalSpace X
                                                                               s : Set (OnePoint X)
                                                                               inst✝² : TopologicalSpace Y
                                                                               inst✝¹ : T2Space Y
                                                                               inst✝ : CompactSpace Y
                                                                               y : Y
                                                                               f : X → Y
                                                                               hf : Topology.IsEmbedding f
                                                                               hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
                                                                               _i : T2Space X
                                                                               this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
                                                                               q : Y
                                                                               hq : Not (Eq q y)
                                                                               ⊢ Membership.mem (Set.range f) q
                                                                             -/
      invFun := fun q ↦ if hq : q = y then ∞ else ↑(show q ∈ range f from by simpa [hy]).choose
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
      left_inv := fun p ↦ by
        /-
          X : Type u_1
          Y : Type u_2
          inst✝³ : TopologicalSpace X
          s : Set (OnePoint X)
          inst✝² : TopologicalSpace Y
          inst✝¹ : T2Space Y
          inst✝ : CompactSpace Y
          y : Y
          f : X → Y
          hf : Topology.IsEmbedding f
          hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
          _i : T2Space X
          this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
          p : OnePoint X
          ⊢ Eq ((fun q => dite (Eq q y) (fun hq => OnePoint.infty) fun hq => ↑(Exists.ch …
        -/
        induction' p using OnePoint.rec with p
          /-
            case infty
            X : Type u_1
            Y : Type u_2
            inst✝³ : TopologicalSpace X
            s : Set (OnePoint X)
            inst✝² : TopologicalSpace Y
            inst✝¹ : T2Space Y
            inst✝ : CompactSpace Y
            y : Y
            f : X → Y
            hf : Topology.IsEmbedding f
            hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
            _i : T2Space X
            this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
            ⊢ Eq ((fun q => dite (Eq q y) (fun hq => OnePoint.infty) fun hq => ↑(Exists.ch …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case coe
            X : Type u_1
            Y : Type u_2
            inst✝³ : TopologicalSpace X
            s : Set (OnePoint X)
            inst✝² : TopologicalSpace Y
            inst✝¹ : T2Space Y
            inst✝ : CompactSpace Y
            y : Y
            f : X → Y
            hf : Topology.IsEmbedding f
            hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
            _i : T2Space X
            this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
            p : X
            ⊢ Eq ((fun q => dite (Eq q y) (fun hq => OnePoint.infty) fun hq => ↑(Exists.ch …
          -/
        · have hp : f p ≠ y := by simpa [hy] using mem_range_self (f := f) p
          /-
            case coe
            X : Type u_1
            Y : Type u_2
            inst✝³ : TopologicalSpace X
            s : Set (OnePoint X)
            inst✝² : TopologicalSpace Y
            inst✝¹ : T2Space Y
            inst✝ : CompactSpace Y
            y : Y
            f : X → Y
            hf : Topology.IsEmbedding f
            hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
            _i : T2Space X
            this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
            p : X
            hp : Ne (f p) y
            ⊢ Eq ((fun q => dite (Eq q y) (fun hq => OnePoint.infty) fun hq => ↑(Exists.ch …
          -/
          simpa [hp] using hf.injective (mem_range_self p).choose_spec
          /-
            🎉 no goals
          -/
      right_inv := fun q ↦ by
        /-
          X : Type u_1
          Y : Type u_2
          inst✝³ : TopologicalSpace X
          s : Set (OnePoint X)
          inst✝² : TopologicalSpace Y
          inst✝¹ : T2Space Y
          inst✝ : CompactSpace Y
          y : Y
          f : X → Y
          hf : Topology.IsEmbedding f
          hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
          _i : T2Space X
          this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
          q : Y
          ⊢ Eq ((fun p => p.elim y f) ((fun q => dite (Eq q y) (fun hq => OnePoint.infty …
        -/
        rcases eq_or_ne q y with rfl | hq
          /-
            case inl
            X : Type u_1
            Y : Type u_2
            inst✝³ : TopologicalSpace X
            s : Set (OnePoint X)
            inst✝² : TopologicalSpace Y
            inst✝¹ : T2Space Y
            inst✝ : CompactSpace Y
            f : X → Y
            hf : Topology.IsEmbedding f
            _i : T2Space X
            q : Y
            hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton q))
            this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds q)
            ⊢ Eq ((fun p => p.elim q f) ((fun q_1 => dite (Eq q_1 q) (fun hq => OnePoint.i …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case inr
            X : Type u_1
            Y : Type u_2
            inst✝³ : TopologicalSpace X
            s : Set (OnePoint X)
            inst✝² : TopologicalSpace Y
            inst✝¹ : T2Space Y
            inst✝ : CompactSpace Y
            y : Y
            f : X → Y
            hf : Topology.IsEmbedding f
            hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
            _i : T2Space X
            this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
            q : Y
            hq : Ne q y
            ⊢ Eq ((fun p => p.elim y f) ((fun q => dite (Eq q y) (fun hq => OnePoint.infty …
          -/
        · have hq' : q ∈ range f := by simpa [hy]
          /-
            case inr
            X : Type u_1
            Y : Type u_2
            inst✝³ : TopologicalSpace X
            s : Set (OnePoint X)
            inst✝² : TopologicalSpace Y
            inst✝¹ : T2Space Y
            inst✝ : CompactSpace Y
            y : Y
            f : X → Y
            hf : Topology.IsEmbedding f
            hy : Eq (Set.range f) (HasCompl.compl (Singleton.singleton y))
            _i : T2Space X
            this : Filter.Tendsto f (Filter.coclosedCompact X) (nhds y)
            q : Y
            hq : Ne q y
            hq' : Membership.mem (Set.range f) q
            ⊢ Eq ((fun p => p.elim y f) ((fun q => dite (Eq q y) (fun hq => OnePoint.infty …
          -/
          simpa [hq] using hq'.choose_spec }
          /-
            🎉 no goals
          -/
  Continuous.homeoOfEquivCompactToT2 <| (continuous_iff e).mpr ⟨this, hf.continuous⟩


@[simp]
lemma equivOfIsEmbeddingOfRangeEq_apply_coe (x : X) :
    equivOfIsEmbeddingOfRangeEq y f hf hy x = f x :=
  rfl


@[simp]
lemma equivOfIsEmbeddingOfRangeEq_apply_infty :
    equivOfIsEmbeddingOfRangeEq y f hf hy ∞ = y :=
  rfl


/-- Extend a homeomorphism of topological spaces
to the homeomorphism of their one point compactifications. -/
@[simps]
def onePointCongr (h : X ≃ₜ Y) : OnePoint X ≃ₜ OnePoint Y where
  __ := h.toEquiv.optionCongr
  toFun := OnePoint.map h
  invFun := OnePoint.map h.symm
  continuous_toFun := continuous_map (map_continuous h) h.map_coclosedCompact.le
  continuous_invFun := continuous_map (map_continuous h.symm) h.symm.map_coclosedCompact.le


/-- A concrete counterexample shows that `Continuous.homeoOfEquivCompactToT2`
cannot be generalized from `T2Space` to `T1Space`.

Let `α = OnePoint ℕ` be the one-point compactification of `ℕ`, and let `β` be the same space
`OnePoint ℕ` with the cofinite topology.  Then `α` is compact, `β` is T1, and the identity map
`id : α → β` is a continuous equivalence that is not a homeomorphism.
-/
theorem Continuous.homeoOfEquivCompactToT2.t1_counterexample :
    ∃ (α β : Type) (_ : TopologicalSpace α) (_ : TopologicalSpace β),
      CompactSpace α ∧ T1Space β ∧ ∃ f : α ≃ β, Continuous f ∧ ¬Continuous f.symm :=
  ⟨OnePoint ℕ, CofiniteTopology (OnePoint ℕ), inferInstance, inferInstance, inferInstance,
    inferInstance, CofiniteTopology.of, CofiniteTopology.continuous_of,
    OnePoint.not_continuous_cofiniteTopology_of_symm⟩

