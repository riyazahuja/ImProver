/-- A function `f : X → E` is *locally integrable on s*, for `s ⊆ X`, if for every `x ∈ s` there is
a neighbourhood of `x` within `s` on which `f` is integrable. (Note this is, in general, strictly
weaker than local integrability with respect to `μ.restrict s`.) -/
def LocallyIntegrableOn (f : X → E) (s : Set X) (μ : Measure X := by volume_tac) : Prop :=
  ∀ x : X, x ∈ s → IntegrableAtFilter f (𝓝[s] x) μ


theorem LocallyIntegrableOn.mono_set (hf : LocallyIntegrableOn f s μ) {t : Set X}
    (hst : t ⊆ s) : LocallyIntegrableOn f t μ := fun x hx =>
  (hf x <| hst hx).filter_mono (nhdsWithin_mono x hst)


theorem LocallyIntegrableOn.norm (hf : LocallyIntegrableOn f s μ) :
    LocallyIntegrableOn (fun x => ‖f x‖) s μ := fun t ht =>
  let ⟨U, hU_nhd, hU_int⟩ := hf t ht
  ⟨U, hU_nhd, hU_int.norm⟩


theorem LocallyIntegrableOn.mono (hf : LocallyIntegrableOn f s μ) {g : X → F}
    (hg : AEStronglyMeasurable g μ) (h : ∀ᵐ x ∂μ, ‖g x‖ ≤ ‖f x‖) :
    LocallyIntegrableOn g s μ := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    g : X → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    h : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (Norm.norm (f x))) (Me …
    ⊢ MeasureTheory.LocallyIntegrableOn g s μ
  -/
  intro x hx
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    g : X → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    h : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (Norm.norm (f x))) (Me …
    x : X
    hx : Membership.mem s x
    ⊢ MeasureTheory.IntegrableAtFilter g (nhdsWithin x s) μ
  -/
  rcases hf x hx with ⟨t, t_mem, ht⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    g : X → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    h : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (Norm.norm (f x))) (Me …
    x : X
    hx : Membership.mem s x
    t : Set X
    t_mem : Membership.mem (nhdsWithin x s) t
    ht : MeasureTheory.IntegrableOn f t μ
    ⊢ MeasureTheory.IntegrableAtFilter g (nhdsWithin x s) μ
  -/
  exact ⟨t, t_mem, Integrable.mono ht hg.restrict (ae_restrict_of_ae h)⟩
  /-
    🎉 no goals
  -/


theorem IntegrableOn.locallyIntegrableOn (hf : IntegrableOn f s μ) : LocallyIntegrableOn f s μ :=
  fun _ _ => ⟨s, self_mem_nhdsWithin, hf⟩


/-- If a function is locally integrable on a compact set, then it is integrable on that set. -/
theorem LocallyIntegrableOn.integrableOn_isCompact (hf : LocallyIntegrableOn f s μ)
    (hs : IsCompact s) : IntegrableOn f s μ :=
  IsCompact.induction_on hs integrableOn_empty (fun _u _v huv hv => hv.mono_set huv)
    (fun _u _v hu hv => integrableOn_union.mpr ⟨hu, hv⟩) hf


theorem LocallyIntegrableOn.integrableOn_compact_subset (hf : LocallyIntegrableOn f s μ) {t : Set X}
    (hst : t ⊆ s) (ht : IsCompact t) : IntegrableOn f t μ :=
  (hf.mono_set hst).integrableOn_isCompact ht


/-- If a function `f` is locally integrable on a set `s` in a second countable topological space,
then there exist countably many open sets `u` covering `s` such that `f` is integrable on each
set `u ∩ s`. -/
theorem LocallyIntegrableOn.exists_countable_integrableOn [SecondCountableTopology X]
    (hf : LocallyIntegrableOn f s μ) : ∃ T : Set (Set X), T.Countable ∧
    (∀ u ∈ T, IsOpen u) ∧ (s ⊆ ⋃ u ∈ T, u) ∧ (∀ u ∈ T, IntegrableOn f (u ∩ s) μ) := by
  have : ∀ x : s, ∃ u, IsOpen u ∧ x.1 ∈ u ∧ IntegrableOn f (u ∩ s) μ := by
    rintro ⟨x, hx⟩
    rcases hf x hx with ⟨t, ht, h't⟩
    rcases mem_nhdsWithin.1 ht with ⟨u, u_open, x_mem, u_sub⟩
    exact ⟨u, u_open, x_mem, h't.mono_set u_sub⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    this : ∀ (x : ↑s), Exists fun u => And (IsOpen u) (And (Membership.mem u ↑x) ( …
    ⊢ Exists fun T => And T.Countable (And (∀ (u : Set X), Membership.mem T u → Is …
  -/
  choose u u_open xu hu using this
  obtain ⟨T, T_count, hT⟩ : ∃ T : Set s, T.Countable ∧ s ⊆ ⋃ i ∈ T, u i := by
    have : s ⊆ ⋃ x : s, u x := fun y hy => mem_iUnion_of_mem ⟨y, hy⟩ (xu ⟨y, hy⟩)
    obtain ⟨T, hT_count, hT_un⟩ := isOpen_iUnion_countable u u_open
    exact ⟨T, hT_count, by rwa [hT_un]⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    u : ↑s → Set X
    u_open : ∀ (x : ↑s), IsOpen (u x)
    xu : ∀ (x : ↑s), Membership.mem (u x) ↑x
    hu : ∀ (x : ↑s), MeasureTheory.IntegrableOn f (Inter.inter (u x) s) μ
    T : Set ↑s
    T_count : T.Countable
    hT : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => u i)
    ⊢ Exists fun T => And T.Countable (And (∀ (u : Set X), Membership.mem T u → Is …
  -/
  refine ⟨u '' T, T_count.image _, ?_, by rwa [biUnion_image], ?_⟩
    /-
      case intro.intro.refine_1
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      u : ↑s → Set X
      u_open : ∀ (x : ↑s), IsOpen (u x)
      xu : ∀ (x : ↑s), Membership.mem (u x) ↑x
      hu : ∀ (x : ↑s), MeasureTheory.IntegrableOn f (Inter.inter (u x) s) μ
      T : Set ↑s
      T_count : T.Countable
      hT : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => u i)
      ⊢ ∀ (u_1 : Set X), Membership.mem (Set.image u T) u_1 → IsOpen u_1
    -/
  · rintro v ⟨w, -, rfl⟩
    /-
      case intro.intro.refine_1.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      u : ↑s → Set X
      u_open : ∀ (x : ↑s), IsOpen (u x)
      xu : ∀ (x : ↑s), Membership.mem (u x) ↑x
      hu : ∀ (x : ↑s), MeasureTheory.IntegrableOn f (Inter.inter (u x) s) μ
      T : Set ↑s
      T_count : T.Countable
      hT : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => u i)
      w : ↑s
      ⊢ IsOpen (u w)
    -/
    exact u_open _
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      u : ↑s → Set X
      u_open : ∀ (x : ↑s), IsOpen (u x)
      xu : ∀ (x : ↑s), Membership.mem (u x) ↑x
      hu : ∀ (x : ↑s), MeasureTheory.IntegrableOn f (Inter.inter (u x) s) μ
      T : Set ↑s
      T_count : T.Countable
      hT : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => u i)
      ⊢ ∀ (u_1 : Set X), Membership.mem (Set.image u T) u_1 → MeasureTheory.Integrab …
    -/
  · rintro v ⟨w, -, rfl⟩
    /-
      case intro.intro.refine_2.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      u : ↑s → Set X
      u_open : ∀ (x : ↑s), IsOpen (u x)
      xu : ∀ (x : ↑s), Membership.mem (u x) ↑x
      hu : ∀ (x : ↑s), MeasureTheory.IntegrableOn f (Inter.inter (u x) s) μ
      T : Set ↑s
      T_count : T.Countable
      hT : HasSubset.Subset s (Set.iUnion fun i => Set.iUnion fun h => u i)
      w : ↑s
      ⊢ MeasureTheory.IntegrableOn f (Inter.inter (u w) s) μ
    -/
    exact hu _
    /-
      🎉 no goals
    -/


/-- If a function `f` is locally integrable on a set `s` in a second countable topological space,
then there exists a sequence of open sets `u n` covering `s` such that `f` is integrable on each
set `u n ∩ s`. -/
theorem LocallyIntegrableOn.exists_nat_integrableOn [SecondCountableTopology X]
    (hf : LocallyIntegrableOn f s μ) : ∃ u : ℕ → Set X,
    (∀ n, IsOpen (u n)) ∧ (s ⊆ ⋃ n, u n) ∧ (∀ n, IntegrableOn f (u n ∩ s) μ) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (HasSubset.Subset s (Se …
  -/
  rcases hf.exists_countable_integrableOn with ⟨T, T_count, T_open, sT, hT⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    T : Set (Set X)
    T_count : T.Countable
    T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
    sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
    hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (HasSubset.Subset s (Se …
  -/
  let T' : Set (Set X) := insert ∅ T
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    T : Set (Set X)
    T_count : T.Countable
    T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
    sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
    hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
    T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (HasSubset.Subset s (Se …
  -/
  have T'_count : T'.Countable := Countable.insert ∅ T_count
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    T : Set (Set X)
    T_count : T.Countable
    T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
    sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
    hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
    T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
    T'_count : T'.Countable
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (HasSubset.Subset s (Se …
  -/
  have T'_ne : T'.Nonempty := by simp only [T', insert_nonempty]
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    T : Set (Set X)
    T_count : T.Countable
    T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
    sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
    hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
    T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
    T'_count : T'.Countable
    T'_ne : T'.Nonempty
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (HasSubset.Subset s (Se …
  -/
  rcases T'_count.exists_eq_range T'_ne with ⟨u, hu⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    T : Set (Set X)
    T_count : T.Countable
    T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
    sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
    hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
    T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
    T'_count : T'.Countable
    T'_ne : T'.Nonempty
    u : Nat → Set X
    hu : Eq T' (Set.range u)
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (HasSubset.Subset s (Se …
  -/
  refine ⟨u, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      ⊢ ∀ (n : Nat), IsOpen (u n)
    -/
  · intro n
    /-
      case intro.intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      n : Nat
      ⊢ IsOpen (u n)
    -/
    have : u n ∈ T' := by rw [hu]; exact mem_range_self n
    /-
      case intro.intro.intro.intro.intro.refine_1
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      n : Nat
      this : Membership.mem T' (u n)
      ⊢ IsOpen (u n)
    -/
    rcases mem_insert_iff.1 this with h|h
      /-
        case intro.intro.intro.intro.intro.refine_1.inl
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : TopologicalSpace X
        inst✝¹ : NormedAddCommGroup E
        f : X → E
        μ : MeasureTheory.Measure X
        s : Set X
        inst✝ : SecondCountableTopology X
        hf : MeasureTheory.LocallyIntegrableOn f s μ
        T : Set (Set X)
        T_count : T.Countable
        T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
        sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
        hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
        T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
        T'_count : T'.Countable
        T'_ne : T'.Nonempty
        u : Nat → Set X
        hu : Eq T' (Set.range u)
        n : Nat
        this : Membership.mem T' (u n)
        h : Eq (u n) EmptyCollection.emptyCollection
        ⊢ IsOpen (u n)
      -/
    · rw [h]
      /-
        case intro.intro.intro.intro.intro.refine_1.inl
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : TopologicalSpace X
        inst✝¹ : NormedAddCommGroup E
        f : X → E
        μ : MeasureTheory.Measure X
        s : Set X
        inst✝ : SecondCountableTopology X
        hf : MeasureTheory.LocallyIntegrableOn f s μ
        T : Set (Set X)
        T_count : T.Countable
        T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
        sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
        hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
        T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
        T'_count : T'.Countable
        T'_ne : T'.Nonempty
        u : Nat → Set X
        hu : Eq T' (Set.range u)
        n : Nat
        this : Membership.mem T' (u n)
        h : Eq (u n) EmptyCollection.emptyCollection
        ⊢ IsOpen EmptyCollection.emptyCollection
      -/
      exact isOpen_empty
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.refine_1.inr
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : TopologicalSpace X
        inst✝¹ : NormedAddCommGroup E
        f : X → E
        μ : MeasureTheory.Measure X
        s : Set X
        inst✝ : SecondCountableTopology X
        hf : MeasureTheory.LocallyIntegrableOn f s μ
        T : Set (Set X)
        T_count : T.Countable
        T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
        sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
        hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
        T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
        T'_count : T'.Countable
        T'_ne : T'.Nonempty
        u : Nat → Set X
        hu : Eq T' (Set.range u)
        n : Nat
        this : Membership.mem T' (u n)
        h : Membership.mem T (u n)
        ⊢ IsOpen (u n)
      -/
    · exact T_open _ h
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      ⊢ HasSubset.Subset s (Set.iUnion fun n => u n)
    -/
  · intro x hx
    /-
      case intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      x : X
      hx : Membership.mem s x
      ⊢ Membership.mem (Set.iUnion fun n => u n) x
    -/
    obtain ⟨v, hv, h'v⟩ : ∃ v, v ∈ T ∧ x ∈ v := by simpa only [mem_iUnion, exists_prop] using sT hx
    /-
      case intro.intro.intro.intro.intro.refine_2.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      x : X
      hx : Membership.mem s x
      v : Set X
      hv : Membership.mem T v
      h'v : Membership.mem v x
      ⊢ Membership.mem (Set.iUnion fun n => u n) x
    -/
    have : v ∈ range u := by rw [← hu]; exact subset_insert ∅ T hv
    /-
      case intro.intro.intro.intro.intro.refine_2.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      x : X
      hx : Membership.mem s x
      v : Set X
      hv : Membership.mem T v
      h'v : Membership.mem v x
      this : Membership.mem (Set.range u) v
      ⊢ Membership.mem (Set.iUnion fun n => u n) x
    -/
    obtain ⟨n, rfl⟩ : ∃ n, u n = v := by simpa only [mem_range] using this
    /-
      case intro.intro.intro.intro.intro.refine_2.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      x : X
      hx : Membership.mem s x
      n : Nat
      hv : Membership.mem T (u n)
      h'v : Membership.mem (u n) x
      this : Membership.mem (Set.range u) (u n)
      ⊢ Membership.mem (Set.iUnion fun n => u n) x
    -/
    exact mem_iUnion_of_mem _ h'v
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_3
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      ⊢ ∀ (n : Nat), MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
    -/
  · intro n
    /-
      case intro.intro.intro.intro.intro.refine_3
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      n : Nat
      ⊢ MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
    -/
    have : u n ∈ T' := by rw [hu]; exact mem_range_self n
    /-
      case intro.intro.intro.intro.intro.refine_3
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : SecondCountableTopology X
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      T : Set (Set X)
      T_count : T.Countable
      T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
      sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
      hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
      T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
      T'_count : T'.Countable
      T'_ne : T'.Nonempty
      u : Nat → Set X
      hu : Eq T' (Set.range u)
      n : Nat
      this : Membership.mem T' (u n)
      ⊢ MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
    -/
    rcases mem_insert_iff.1 this with h|h
      /-
        case intro.intro.intro.intro.intro.refine_3.inl
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : TopologicalSpace X
        inst✝¹ : NormedAddCommGroup E
        f : X → E
        μ : MeasureTheory.Measure X
        s : Set X
        inst✝ : SecondCountableTopology X
        hf : MeasureTheory.LocallyIntegrableOn f s μ
        T : Set (Set X)
        T_count : T.Countable
        T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
        sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
        hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
        T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
        T'_count : T'.Countable
        T'_ne : T'.Nonempty
        u : Nat → Set X
        hu : Eq T' (Set.range u)
        n : Nat
        this : Membership.mem T' (u n)
        h : Eq (u n) EmptyCollection.emptyCollection
        ⊢ MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
      -/
    · simp only [h, empty_inter, integrableOn_empty]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.refine_3.inr
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : TopologicalSpace X
        inst✝¹ : NormedAddCommGroup E
        f : X → E
        μ : MeasureTheory.Measure X
        s : Set X
        inst✝ : SecondCountableTopology X
        hf : MeasureTheory.LocallyIntegrableOn f s μ
        T : Set (Set X)
        T_count : T.Countable
        T_open : ∀ (u : Set X), Membership.mem T u → IsOpen u
        sT : HasSubset.Subset s (Set.iUnion fun u => Set.iUnion fun h => u)
        hT : ∀ (u : Set X), Membership.mem T u → MeasureTheory.IntegrableOn f (Inter.i …
        T' : Set (Set X) := Insert.insert EmptyCollection.emptyCollection T
        T'_count : T'.Countable
        T'_ne : T'.Nonempty
        u : Nat → Set X
        hu : Eq T' (Set.range u)
        n : Nat
        this : Membership.mem T' (u n)
        h : Membership.mem T (u n)
        ⊢ MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
      -/
    · exact hT _ h
      /-
        🎉 no goals
      -/


theorem LocallyIntegrableOn.aestronglyMeasurable [SecondCountableTopology X]
    (hf : LocallyIntegrableOn f s μ) : AEStronglyMeasurable f (μ.restrict s) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  rcases hf.exists_nat_integrableOn with ⟨u, -, su, hu⟩
  /-
    case intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    u : Nat → Set X
    su : HasSubset.Subset s (Set.iUnion fun n => u n)
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  have : s = ⋃ n, u n ∩ s := by rw [← iUnion_inter]; exact (inter_eq_right.mpr su).symm
  /-
    case intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    u : Nat → Set X
    su : HasSubset.Subset s (Set.iUnion fun n => u n)
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
    this : Eq s (Set.iUnion fun n => Inter.inter (u n) s)
    ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
  -/
  rw [this, aestronglyMeasurable_iUnion_iff]
  /-
    case intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    u : Nat → Set X
    su : HasSubset.Subset s (Set.iUnion fun n => u n)
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (Inter.inter (u n) s) μ
    this : Eq s (Set.iUnion fun n => Inter.inter (u n) s)
    ⊢ ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable f (μ.restrict (Inter.inter ( …
  -/
  exact fun i : ℕ => (hu i).aestronglyMeasurable
  /-
    🎉 no goals
  -/


/-- If `s` is locally closed (e.g. open or closed), then `f` is locally integrable on `s` iff it is
integrable on every compact subset contained in `s`. -/
theorem locallyIntegrableOn_iff [LocallyCompactSpace X] (hs : IsLocallyClosed s) :
    LocallyIntegrableOn f s μ ↔ ∀ (k : Set X), k ⊆ s → IsCompact k → IntegrableOn f k μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : LocallyCompactSpace X
    hs : IsLocallyClosed s
    ⊢ Iff (MeasureTheory.LocallyIntegrableOn f s μ) (∀ (k : Set X), HasSubset.Subs …
  -/
  refine ⟨fun hf k hk ↦ hf.integrableOn_compact_subset hk, fun hf x hx ↦ ?_⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : LocallyCompactSpace X
    hs : IsLocallyClosed s
    hf : ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integra …
    x : X
    hx : Membership.mem s x
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x s) μ
  -/
  rcases hs with ⟨U, Z, hU, hZ, rfl⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : LocallyCompactSpace X
    x : X
    U Z : Set X
    hU : IsOpen U
    hZ : IsClosed Z
    hf : ∀ (k : Set X), HasSubset.Subset k (Inter.inter U Z) → IsCompact k → Measu …
    hx : Membership.mem (Inter.inter U Z) x
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x (Inter.inter U Z)) μ
  -/
  rcases exists_compact_subset hU hx.1 with ⟨K, hK, hxK, hKU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : LocallyCompactSpace X
    x : X
    U Z : Set X
    hU : IsOpen U
    hZ : IsClosed Z
    hf : ∀ (k : Set X), HasSubset.Subset k (Inter.inter U Z) → IsCompact k → Measu …
    hx : Membership.mem (Inter.inter U Z) x
    K : Set X
    hK : IsCompact K
    hxK : Membership.mem (interior K) x
    hKU : HasSubset.Subset K U
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x (Inter.inter U Z)) μ
  -/
  rw [nhdsWithin_inter_of_mem (nhdsWithin_le_nhds <| hU.mem_nhds hx.1)]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : LocallyCompactSpace X
    x : X
    U Z : Set X
    hU : IsOpen U
    hZ : IsClosed Z
    hf : ∀ (k : Set X), HasSubset.Subset k (Inter.inter U Z) → IsCompact k → Measu …
    hx : Membership.mem (Inter.inter U Z) x
    K : Set X
    hK : IsCompact K
    hxK : Membership.mem (interior K) x
    hKU : HasSubset.Subset K U
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x Z) μ
  -/
  refine ⟨Z ∩ K, inter_mem_nhdsWithin _ (mem_interior_iff_mem_nhds.1 hxK), ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : LocallyCompactSpace X
    x : X
    U Z : Set X
    hU : IsOpen U
    hZ : IsClosed Z
    hf : ∀ (k : Set X), HasSubset.Subset k (Inter.inter U Z) → IsCompact k → Measu …
    hx : Membership.mem (Inter.inter U Z) x
    K : Set X
    hK : IsCompact K
    hxK : Membership.mem (interior K) x
    hKU : HasSubset.Subset K U
    ⊢ MeasureTheory.IntegrableOn f (Inter.inter Z K) μ
  -/
  exact hf (Z ∩ K) (fun y hy ↦ ⟨hKU hy.2, hy.1⟩) (.inter_left hK hZ)
  /-
    🎉 no goals
  -/


protected theorem LocallyIntegrableOn.add
    (hf : LocallyIntegrableOn f s μ) (hg : LocallyIntegrableOn g s μ) :
    LocallyIntegrableOn (f + g) s μ := fun x hx ↦ (hf x hx).add (hg x hx)


protected theorem LocallyIntegrableOn.sub
    (hf : LocallyIntegrableOn f s μ) (hg : LocallyIntegrableOn g s μ) :
    LocallyIntegrableOn (f - g) s μ := fun x hx ↦ (hf x hx).sub (hg x hx)


protected theorem LocallyIntegrableOn.neg (hf : LocallyIntegrableOn f s μ) :
    LocallyIntegrableOn (-f) s μ := fun x hx ↦ (hf x hx).neg


/-- A function `f : X → E` is *locally integrable* if it is integrable on a neighborhood of every
point. In particular, it is integrable on all compact sets,
see `LocallyIntegrable.integrableOn_isCompact`. -/
def LocallyIntegrable (f : X → E) (μ : Measure X := by volume_tac) : Prop :=
  ∀ x : X, IntegrableAtFilter f (𝓝 x) μ


theorem locallyIntegrable_comap (hs : MeasurableSet s) :
    LocallyIntegrable (fun x : s ↦ f x) (μ.comap Subtype.val) ↔ LocallyIntegrableOn f s μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    hs : MeasurableSet s
    ⊢ Iff (MeasureTheory.LocallyIntegrable (fun x => f ↑x) (MeasureTheory.Measure. …
  -/
  simp_rw [LocallyIntegrableOn, Subtype.forall', ← map_nhds_subtype_val]
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    hs : MeasurableSet s
    ⊢ Iff (MeasureTheory.LocallyIntegrable (fun x => f ↑x) (MeasureTheory.Measure. …
  -/
  exact forall_congr' fun _ ↦ (MeasurableEmbedding.subtype_coe hs).integrableAtFilter_iff_comap.symm
  /-
    🎉 no goals
  -/


theorem locallyIntegrableOn_univ : LocallyIntegrableOn f univ μ ↔ LocallyIntegrable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    ⊢ Iff (MeasureTheory.LocallyIntegrableOn f Set.univ μ) (MeasureTheory.LocallyI …
  -/
  simp only [LocallyIntegrableOn, nhdsWithin_univ, mem_univ, true_imp_iff]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem LocallyIntegrable.locallyIntegrableOn (hf : LocallyIntegrable f μ) (s : Set X) :
    LocallyIntegrableOn f s μ := fun x _ => (hf x).filter_mono nhdsWithin_le_nhds


theorem Integrable.locallyIntegrable (hf : Integrable f μ) : LocallyIntegrable f μ := fun _ =>
  hf.integrableAtFilter _


theorem LocallyIntegrable.mono (hf : LocallyIntegrable f μ) {g : X → F}
    (hg : AEStronglyMeasurable g μ) (h : ∀ᵐ x ∂μ, ‖g x‖ ≤ ‖f x‖) :
    LocallyIntegrable g μ := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : X → E
    μ : MeasureTheory.Measure X
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    h : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (Norm.norm (f x))) (Me …
    ⊢ MeasureTheory.LocallyIntegrable g μ
  -/
  rw [← locallyIntegrableOn_univ] at hf ⊢
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedAddCommGroup F
    f : X → E
    μ : MeasureTheory.Measure X
    hf : MeasureTheory.LocallyIntegrableOn f Set.univ μ
    g : X → F
    hg : MeasureTheory.AEStronglyMeasurable g μ
    h : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (Norm.norm (f x))) (Me …
    ⊢ MeasureTheory.LocallyIntegrableOn g Set.univ μ
  -/
  exact hf.mono hg h
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable with respect to `μ.restrict s`, it is locally integrable on `s`.
(See `locallyIntegrableOn_iff_locallyIntegrable_restrict` for an iff statement when `s` is
closed.) -/
theorem locallyIntegrableOn_of_locallyIntegrable_restrict [OpensMeasurableSpace X]
    (hf : LocallyIntegrable f (μ.restrict s)) : LocallyIntegrableOn f s μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : OpensMeasurableSpace X
    hf : MeasureTheory.LocallyIntegrable f (μ.restrict s)
    ⊢ MeasureTheory.LocallyIntegrableOn f s μ
  -/
  intro x _
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : OpensMeasurableSpace X
    hf : MeasureTheory.LocallyIntegrable f (μ.restrict s)
    x : X
    a✝ : Membership.mem s x
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x s) μ
  -/
  obtain ⟨t, ht_mem, ht_int⟩ := hf x
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : OpensMeasurableSpace X
    hf : MeasureTheory.LocallyIntegrable f (μ.restrict s)
    x : X
    a✝ : Membership.mem s x
    t : Set X
    ht_mem : Membership.mem (nhds x) t
    ht_int : MeasureTheory.IntegrableOn f t (μ.restrict s)
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x s) μ
  -/
  obtain ⟨u, hu_sub, hu_o, hu_mem⟩ := mem_nhds_iff.mp ht_mem
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : OpensMeasurableSpace X
    hf : MeasureTheory.LocallyIntegrable f (μ.restrict s)
    x : X
    a✝ : Membership.mem s x
    t : Set X
    ht_mem : Membership.mem (nhds x) t
    ht_int : MeasureTheory.IntegrableOn f t (μ.restrict s)
    u : Set X
    hu_sub : HasSubset.Subset u t
    hu_o : IsOpen u
    hu_mem : Membership.mem u x
    ⊢ MeasureTheory.IntegrableAtFilter f (nhdsWithin x s) μ
  -/
  refine ⟨_, inter_mem_nhdsWithin s (hu_o.mem_nhds hu_mem), ?_⟩
  simpa only [IntegrableOn, Measure.restrict_restrict hu_o.measurableSet, inter_comm] using
    ht_int.mono_set hu_sub


/-- If `s` is closed, being locally integrable on `s` wrt `μ` is equivalent to being locally
integrable with respect to `μ.restrict s`. For the one-way implication without assuming `s` closed,
see `locallyIntegrableOn_of_locallyIntegrable_restrict`. -/
theorem locallyIntegrableOn_iff_locallyIntegrable_restrict [OpensMeasurableSpace X]
    (hs : IsClosed s) : LocallyIntegrableOn f s μ ↔ LocallyIntegrable f (μ.restrict s) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : OpensMeasurableSpace X
    hs : IsClosed s
    ⊢ Iff (MeasureTheory.LocallyIntegrableOn f s μ) (MeasureTheory.LocallyIntegrab …
  -/
  refine ⟨fun hf x => ?_, locallyIntegrableOn_of_locallyIntegrable_restrict⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝ : OpensMeasurableSpace X
    hs : IsClosed s
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    x : X
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (μ.restrict s)
  -/
  by_cases h : x ∈ s
    /-
      case pos
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsClosed s
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Membership.mem s x
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (μ.restrict s)
    -/
  · obtain ⟨t, ht_nhds, ht_int⟩ := hf x h
    /-
      case pos.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsClosed s
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Membership.mem s x
      t : Set X
      ht_nhds : Membership.mem (nhdsWithin x s) t
      ht_int : MeasureTheory.IntegrableOn f t μ
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (μ.restrict s)
    -/
    obtain ⟨u, hu_o, hu_x, hu_sub⟩ := mem_nhdsWithin.mp ht_nhds
    /-
      case pos.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsClosed s
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Membership.mem s x
      t : Set X
      ht_nhds : Membership.mem (nhdsWithin x s) t
      ht_int : MeasureTheory.IntegrableOn f t μ
      u : Set X
      hu_o : IsOpen u
      hu_x : Membership.mem u x
      hu_sub : HasSubset.Subset (Inter.inter u s) t
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (μ.restrict s)
    -/
    refine ⟨u, hu_o.mem_nhds hu_x, ?_⟩
    /-
      case pos.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsClosed s
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Membership.mem s x
      t : Set X
      ht_nhds : Membership.mem (nhdsWithin x s) t
      ht_int : MeasureTheory.IntegrableOn f t μ
      u : Set X
      hu_o : IsOpen u
      hu_x : Membership.mem u x
      hu_sub : HasSubset.Subset (Inter.inter u s) t
      ⊢ MeasureTheory.IntegrableOn f u (μ.restrict s)
    -/
    rw [IntegrableOn, restrict_restrict hu_o.measurableSet]
    /-
      case pos.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsClosed s
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Membership.mem s x
      t : Set X
      ht_nhds : Membership.mem (nhdsWithin x s) t
      ht_int : MeasureTheory.IntegrableOn f t μ
      u : Set X
      hu_o : IsOpen u
      hu_x : Membership.mem u x
      hu_sub : HasSubset.Subset (Inter.inter u s) t
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter u s))
    -/
    exact ht_int.mono_set hu_sub
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsClosed s
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Not (Membership.mem s x)
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (μ.restrict s)
    -/
  · rw [← isOpen_compl_iff] at hs
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsOpen (HasCompl.compl s)
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Not (Membership.mem s x)
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (μ.restrict s)
    -/
    refine ⟨sᶜ, hs.mem_nhds h, ?_⟩
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsOpen (HasCompl.compl s)
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Not (Membership.mem s x)
      ⊢ MeasureTheory.IntegrableOn f (HasCompl.compl s) (μ.restrict s)
    -/
    rw [IntegrableOn, restrict_restrict, inter_comm, inter_compl_self, ← IntegrableOn]
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : TopologicalSpace X
      inst✝¹ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      s : Set X
      inst✝ : OpensMeasurableSpace X
      hs : IsOpen (HasCompl.compl s)
      hf : MeasureTheory.LocallyIntegrableOn f s μ
      x : X
      h : Not (Membership.mem s x)
      ⊢ MeasureTheory.IntegrableOn f EmptyCollection.emptyCollection μ
    -/
    exacts [integrableOn_empty, hs.measurableSet]
    /-
      🎉 no goals
    -/


/-- If a function is locally integrable, then it is integrable on any compact set. -/
theorem LocallyIntegrable.integrableOn_isCompact {k : Set X} (hf : LocallyIntegrable f μ)
    (hk : IsCompact k) : IntegrableOn f k μ :=
  (hf.locallyIntegrableOn k).integrableOn_isCompact hk


/-- If a function is locally integrable, then it is integrable on an open neighborhood of any
compact set. -/
theorem LocallyIntegrable.integrableOn_nhds_isCompact (hf : LocallyIntegrable f μ) {k : Set X}
    (hk : IsCompact k) : ∃ u, IsOpen u ∧ k ⊆ u ∧ IntegrableOn f u μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    hf : MeasureTheory.LocallyIntegrable f μ
    k : Set X
    hk : IsCompact k
    ⊢ Exists fun u => And (IsOpen u) (And (HasSubset.Subset k u) (MeasureTheory.In …
  -/
  refine IsCompact.induction_on hk ?_ ?_ ?_ ?_
    /-
      case refine_1
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      ⊢ Exists fun u => And (IsOpen u) (And (HasSubset.Subset EmptyCollection.emptyC …
    -/
  · refine ⟨∅, isOpen_empty, Subset.rfl, integrableOn_empty⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      ⊢ ∀ ⦃s t : Set X⦄, HasSubset.Subset s t → (Exists fun u => And (IsOpen u) (And …
    -/
  · rintro s t hst ⟨u, u_open, tu, hu⟩
    /-
      case refine_2.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      s t : Set X
      hst : HasSubset.Subset s t
      u : Set X
      u_open : IsOpen u
      tu : HasSubset.Subset t u
      hu : MeasureTheory.IntegrableOn f u μ
      ⊢ Exists fun u => And (IsOpen u) (And (HasSubset.Subset s u) (MeasureTheory.In …
    -/
    exact ⟨u, u_open, hst.trans tu, hu⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      ⊢ ∀ ⦃s t : Set X⦄, (Exists fun u => And (IsOpen u) (And (HasSubset.Subset s u) …
    -/
  · rintro s t ⟨u, u_open, su, hu⟩ ⟨v, v_open, tv, hv⟩
    /-
      case refine_3.intro.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      s t u : Set X
      u_open : IsOpen u
      su : HasSubset.Subset s u
      hu : MeasureTheory.IntegrableOn f u μ
      v : Set X
      v_open : IsOpen v
      tv : HasSubset.Subset t v
      hv : MeasureTheory.IntegrableOn f v μ
      ⊢ Exists fun u => And (IsOpen u) (And (HasSubset.Subset (Union.union s t) u) ( …
    -/
    exact ⟨u ∪ v, u_open.union v_open, union_subset_union su tv, hu.union hv⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      ⊢ ∀ (x : X), Membership.mem k x → Exists fun t => And (Membership.mem (nhdsWit …
    -/
  · intro x _
    /-
      case refine_4
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      x : X
      a✝ : Membership.mem k x
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x k) t) (Exists fun u => And …
    -/
    rcases hf x with ⟨u, ux, hu⟩
    /-
      case refine_4.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      x : X
      a✝ : Membership.mem k x
      u : Set X
      ux : Membership.mem (nhds x) u
      hu : MeasureTheory.IntegrableOn f u μ
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x k) t) (Exists fun u => And …
    -/
    rcases mem_nhds_iff.1 ux with ⟨v, vu, v_open, xv⟩
    /-
      case refine_4.intro.intro.intro.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : TopologicalSpace X
      inst✝ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      hf : MeasureTheory.LocallyIntegrable f μ
      k : Set X
      hk : IsCompact k
      x : X
      a✝ : Membership.mem k x
      u : Set X
      ux : Membership.mem (nhds x) u
      hu : MeasureTheory.IntegrableOn f u μ
      v : Set X
      vu : HasSubset.Subset v u
      v_open : IsOpen v
      xv : Membership.mem v x
      ⊢ Exists fun t => And (Membership.mem (nhdsWithin x k) t) (Exists fun u => And …
    -/
    exact ⟨v, nhdsWithin_le_nhds (v_open.mem_nhds xv), v, v_open, Subset.rfl, hu.mono_set vu⟩
    /-
      🎉 no goals
    -/


theorem locallyIntegrable_iff [LocallyCompactSpace X] :
    LocallyIntegrable f μ ↔ ∀ k : Set X, IsCompact k → IntegrableOn f k μ :=
  ⟨fun hf _k hk => hf.integrableOn_isCompact hk, fun hf x =>
    let ⟨K, hK, h2K⟩ := exists_compact_mem_nhds x
    ⟨K, h2K, hf K hK⟩⟩


theorem LocallyIntegrable.aestronglyMeasurable [SecondCountableTopology X]
    (hf : LocallyIntegrable f μ) : AEStronglyMeasurable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable f μ
  -/
  simpa only [restrict_univ] using (locallyIntegrableOn_univ.mpr hf).aestronglyMeasurable
  /-
    🎉 no goals
  -/


/-- If a function is locally integrable in a second countable topological space,
then there exists a sequence of open sets covering the space on which it is integrable. -/
theorem LocallyIntegrable.exists_nat_integrableOn [SecondCountableTopology X]
    (hf : LocallyIntegrable f μ) : ∃ u : ℕ → Set X,
    (∀ n, IsOpen (u n)) ∧ ((⋃ n, u n) = univ) ∧ (∀ n, IntegrableOn f (u n) μ) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrable f μ
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (Eq (Set.iUnion fun n = …
  -/
  rcases (hf.locallyIntegrableOn univ).exists_nat_integrableOn with ⟨u, u_open, u_union, hu⟩
  /-
    case intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set X
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_union : HasSubset.Subset Set.univ (Set.iUnion fun n => u n)
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (Inter.inter (u n) Set.univ) μ
    ⊢ Exists fun u => And (∀ (n : Nat), IsOpen (u n)) (And (Eq (Set.iUnion fun n = …
  -/
  refine ⟨u, u_open, eq_univ_of_univ_subset u_union, fun n ↦ ?_⟩
  /-
    case intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝ : SecondCountableTopology X
    hf : MeasureTheory.LocallyIntegrable f μ
    u : Nat → Set X
    u_open : ∀ (n : Nat), IsOpen (u n)
    u_union : HasSubset.Subset Set.univ (Set.iUnion fun n => u n)
    hu : ∀ (n : Nat), MeasureTheory.IntegrableOn f (Inter.inter (u n) Set.univ) μ
    n : Nat
    ⊢ MeasureTheory.IntegrableOn f (u n) μ
  -/
  simpa only [inter_univ] using hu n
  /-
    🎉 no goals
  -/


theorem Memℒp.locallyIntegrable [IsLocallyFiniteMeasure μ] {f : X → E} {p : ℝ≥0∞}
    (hf : Memℒp f p μ) (hp : 1 ≤ p) : LocallyIntegrable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → E
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp : LE.le 1 p
    ⊢ MeasureTheory.LocallyIntegrable f μ
  -/
  intro x
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → E
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp : LE.le 1 p
    x : X
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) μ
  -/
  rcases μ.finiteAt_nhds x with ⟨U, hU, h'U⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → E
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp : LE.le 1 p
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : LT.lt (μ U) Top.top
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) μ
  -/
  have : Fact (μ U < ⊤) := ⟨h'U⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → E
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp : LE.le 1 p
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : LT.lt (μ U) Top.top
    this : Fact (LT.lt (μ U) Top.top)
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) μ
  -/
  refine ⟨U, hU, ?_⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → E
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp : LE.le 1 p
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : LT.lt (μ U) Top.top
    this : Fact (LT.lt (μ U) Top.top)
    ⊢ MeasureTheory.IntegrableOn f U μ
  -/
  rw [IntegrableOn, ← memℒp_one_iff_integrable]
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : TopologicalSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → E
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp : LE.le 1 p
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : LT.lt (μ U) Top.top
    this : Fact (LT.lt (μ U) Top.top)
    ⊢ MeasureTheory.Memℒp f 1 (μ.restrict U)
  -/
  apply (hf.restrict U).memℒp_of_exponent_le hp
  /-
    🎉 no goals
  -/


theorem locallyIntegrable_const [IsLocallyFiniteMeasure μ] (c : E) :
    LocallyIntegrable (fun _ => c) μ :=
  (memℒp_top_const c).locallyIntegrable le_top


theorem locallyIntegrableOn_const [IsLocallyFiniteMeasure μ] (c : E) :
    LocallyIntegrableOn (fun _ => c) s μ :=
  (locallyIntegrable_const c).locallyIntegrableOn s


theorem locallyIntegrable_zero : LocallyIntegrable (fun _ ↦ (0 : E)) μ :=
  (integrable_zero X E μ).locallyIntegrable


theorem locallyIntegrableOn_zero : LocallyIntegrableOn (fun _ ↦ (0 : E)) s μ :=
  locallyIntegrable_zero.locallyIntegrableOn s


theorem LocallyIntegrable.indicator (hf : LocallyIntegrable f μ) {s : Set X}
    (hs : MeasurableSet s) : LocallyIntegrable (s.indicator f) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    hf : MeasureTheory.LocallyIntegrable f μ
    s : Set X
    hs : MeasurableSet s
    ⊢ MeasureTheory.LocallyIntegrable (s.indicator f) μ
  -/
  intro x
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    hf : MeasureTheory.LocallyIntegrable f μ
    s : Set X
    hs : MeasurableSet s
    x : X
    ⊢ MeasureTheory.IntegrableAtFilter (s.indicator f) (nhds x) μ
  -/
  rcases hf x with ⟨U, hU, h'U⟩
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    hf : MeasureTheory.LocallyIntegrable f μ
    s : Set X
    hs : MeasurableSet s
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : MeasureTheory.IntegrableOn f U μ
    ⊢ MeasureTheory.IntegrableAtFilter (s.indicator f) (nhds x) μ
  -/
  exact ⟨U, hU, h'U.indicator hs⟩
  /-
    🎉 no goals
  -/


theorem locallyIntegrable_map_homeomorph [BorelSpace X] [BorelSpace Y] (e : X ≃ₜ Y) {f : Y → E}
    {μ : Measure X} : LocallyIntegrable f (Measure.map e μ) ↔ LocallyIntegrable (f ∘ e) μ := by
  /-
    X : Type u_1
    Y : Type u_2
    E : Type u_3
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : TopologicalSpace Y
    inst✝² : NormedAddCommGroup E
    inst✝¹ : BorelSpace X
    inst✝ : BorelSpace Y
    e : Homeomorph X Y
    f : Y → E
    μ : MeasureTheory.Measure X
    ⊢ Iff (MeasureTheory.LocallyIntegrable f (MeasureTheory.Measure.map (⇑e) μ)) ( …
  -/
  refine ⟨fun h x => ?_, fun h x => ?_⟩
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable f (MeasureTheory.Measure.map (⇑e) μ)
      x : X
      ⊢ MeasureTheory.IntegrableAtFilter (Function.comp f ⇑e) (nhds x) μ
    -/
  · rcases h (e x) with ⟨U, hU, h'U⟩
    /-
      case refine_1.intro.intro
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable f (MeasureTheory.Measure.map (⇑e) μ)
      x : X
      U : Set Y
      hU : Membership.mem (nhds (e x)) U
      h'U : MeasureTheory.IntegrableOn f U (MeasureTheory.Measure.map (⇑e) μ)
      ⊢ MeasureTheory.IntegrableAtFilter (Function.comp f ⇑e) (nhds x) μ
    -/
    refine ⟨e ⁻¹' U, e.continuous.continuousAt.preimage_mem_nhds hU, ?_⟩
    /-
      case refine_1.intro.intro
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable f (MeasureTheory.Measure.map (⇑e) μ)
      x : X
      U : Set Y
      hU : Membership.mem (nhds (e x)) U
      h'U : MeasureTheory.IntegrableOn f U (MeasureTheory.Measure.map (⇑e) μ)
      ⊢ MeasureTheory.IntegrableOn (Function.comp f ⇑e) (Set.preimage (⇑e) U) μ
    -/
    exact (integrableOn_map_equiv e.toMeasurableEquiv).1 h'U
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x : Y
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (MeasureTheory.Measure.map (⇑e) μ)
    -/
  · rcases h (e.symm x) with ⟨U, hU, h'U⟩
    /-
      case refine_2.intro.intro
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x : Y
      U : Set X
      hU : Membership.mem (nhds (e.symm x)) U
      h'U : MeasureTheory.IntegrableOn (Function.comp f ⇑e) U μ
      ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) (MeasureTheory.Measure.map (⇑e) μ)
    -/
    refine ⟨e.symm ⁻¹' U, e.symm.continuous.continuousAt.preimage_mem_nhds hU, ?_⟩
    /-
      case refine_2.intro.intro
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x : Y
      U : Set X
      hU : Membership.mem (nhds (e.symm x)) U
      h'U : MeasureTheory.IntegrableOn (Function.comp f ⇑e) U μ
      ⊢ MeasureTheory.IntegrableOn f (Set.preimage (⇑e.symm) U) (MeasureTheory.Measu …
    -/
    apply (integrableOn_map_equiv e.toMeasurableEquiv).2
    /-
      case refine_2.intro.intro
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x : Y
      U : Set X
      hU : Membership.mem (nhds (e.symm x)) U
      h'U : MeasureTheory.IntegrableOn (Function.comp f ⇑e) U μ
      ⊢ MeasureTheory.IntegrableOn (Function.comp f ⇑e.toMeasurableEquiv) (Set.preim …
    -/
    simp only [Homeomorph.toMeasurableEquiv_coe]
    /-
      case refine_2.intro.intro
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x : Y
      U : Set X
      hU : Membership.mem (nhds (e.symm x)) U
      h'U : MeasureTheory.IntegrableOn (Function.comp f ⇑e) U μ
      ⊢ MeasureTheory.IntegrableOn (Function.comp f ⇑e) (Set.preimage (⇑e) (Set.prei …
    -/
    convert h'U
    /-
      case h.e'_7
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x : Y
      U : Set X
      hU : Membership.mem (nhds (e.symm x)) U
      h'U : MeasureTheory.IntegrableOn (Function.comp f ⇑e) U μ
      ⊢ Eq (Set.preimage (⇑e) (Set.preimage (⇑e.symm) U)) U
    -/
    ext x
    /-
      case h.e'_7.h
      X : Type u_1
      Y : Type u_2
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : MeasurableSpace Y
      inst✝³ : TopologicalSpace Y
      inst✝² : NormedAddCommGroup E
      inst✝¹ : BorelSpace X
      inst✝ : BorelSpace Y
      e : Homeomorph X Y
      f : Y → E
      μ : MeasureTheory.Measure X
      h : MeasureTheory.LocallyIntegrable (Function.comp f ⇑e) μ
      x✝ : Y
      U : Set X
      hU : Membership.mem (nhds (e.symm x✝)) U
      h'U : MeasureTheory.IntegrableOn (Function.comp f ⇑e) U μ
      x : X
      ⊢ Iff (Membership.mem (Set.preimage (⇑e) (Set.preimage (⇑e.symm) U)) x) (Membe …
    -/
    simp only [mem_preimage, Homeomorph.symm_apply_apply]
    /-
      🎉 no goals
    -/


protected theorem LocallyIntegrable.add (hf : LocallyIntegrable f μ) (hg : LocallyIntegrable g μ) :
    LocallyIntegrable (f + g) μ := fun x ↦ (hf x).add (hg x)


protected theorem LocallyIntegrable.sub (hf : LocallyIntegrable f μ) (hg : LocallyIntegrable g μ) :
    LocallyIntegrable (f - g) μ := fun x ↦ (hf x).sub (hg x)


protected theorem LocallyIntegrable.neg (hf : LocallyIntegrable f μ) :
    LocallyIntegrable (-f) μ := fun x ↦ (hf x).neg


protected theorem LocallyIntegrable.smul {𝕜 : Type*} [NormedAddCommGroup 𝕜] [SMulZeroClass 𝕜 E]
    [BoundedSMul 𝕜 E] (hf : LocallyIntegrable f μ) (c : 𝕜) :
    LocallyIntegrable (c • f) μ := fun x ↦ (hf x).smul c


theorem locallyIntegrable_finset_sum' {ι} (s : Finset ι) {f : ι → X → E}
    (hf : ∀ i ∈ s, LocallyIntegrable (f i) μ) : LocallyIntegrable (∑ i ∈ s, f i) μ :=
  Finset.sum_induction f (fun g => LocallyIntegrable g μ) (fun _ _ => LocallyIntegrable.add)
    locallyIntegrable_zero hf


theorem locallyIntegrable_finset_sum {ι} (s : Finset ι) {f : ι → X → E}
    (hf : ∀ i ∈ s, LocallyIntegrable (f i) μ) : LocallyIntegrable (fun a ↦ ∑ i ∈ s, f i a) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    ι : Type u_6
    s : Finset ι
    f : ι → X → E
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.LocallyIntegrable (f i) μ
    ⊢ MeasureTheory.LocallyIntegrable (fun a => s.sum fun i => f i a) μ
  -/
  simpa only [← Finset.sum_apply] using locallyIntegrable_finset_sum' s hf
  /-
    🎉 no goals
  -/


/-- If `f` is locally integrable and `g` is continuous with compact support,
then `g • f` is integrable. -/
theorem LocallyIntegrable.integrable_smul_left_of_hasCompactSupport
    [NormedSpace ℝ E] [OpensMeasurableSpace X] [T2Space X]
    (hf : LocallyIntegrable f μ) {g : X → ℝ} (hg : Continuous g) (h'g : HasCompactSupport g) :
    Integrable (fun x ↦ g x • f x) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → Real
    hg : Continuous g
    h'g : HasCompactSupport g
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (g x) (f x)) μ
  -/
  let K := tsupport g
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → Real
    hg : Continuous g
    h'g : HasCompactSupport g
    K : Set X := tsupport g
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (g x) (f x)) μ
  -/
  have hK : IsCompact K := h'g
  have : K.indicator (fun x ↦ g x • f x) = (fun x ↦ g x • f x) := by
    apply indicator_eq_self.2
    apply support_subset_iff'.2
    intros x hx
    simp [image_eq_zero_of_nmem_tsupport hx]
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → Real
    hg : Continuous g
    h'g : HasCompactSupport g
    K : Set X := tsupport g
    hK : IsCompact K
    this : Eq (K.indicator fun x => HSMul.hSMul (g x) (f x)) fun x => HSMul.hSMul  …
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (g x) (f x)) μ
  -/
  rw [← this, indicator_smul]
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → Real
    hg : Continuous g
    h'g : HasCompactSupport g
    K : Set X := tsupport g
    hK : IsCompact K
    this : Eq (K.indicator fun x => HSMul.hSMul (g x) (f x)) fun x => HSMul.hSMul  …
    ⊢ MeasureTheory.Integrable (fun a => HSMul.hSMul (g a) (K.indicator f a)) μ
  -/
  apply Integrable.smul_of_top_right
    /-
      case hf
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝² : NormedSpace Real E
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : T2Space X
      hf : MeasureTheory.LocallyIntegrable f μ
      g : X → Real
      hg : Continuous g
      h'g : HasCompactSupport g
      K : Set X := tsupport g
      hK : IsCompact K
      this : Eq (K.indicator fun x => HSMul.hSMul (g x) (f x)) fun x => HSMul.hSMul  …
      ⊢ MeasureTheory.Integrable (K.indicator f) μ
    -/
  · rw [integrable_indicator_iff hK.measurableSet]
    /-
      case hf
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝² : NormedSpace Real E
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : T2Space X
      hf : MeasureTheory.LocallyIntegrable f μ
      g : X → Real
      hg : Continuous g
      h'g : HasCompactSupport g
      K : Set X := tsupport g
      hK : IsCompact K
      this : Eq (K.indicator fun x => HSMul.hSMul (g x) (f x)) fun x => HSMul.hSMul  …
      ⊢ MeasureTheory.IntegrableOn f K μ
    -/
    exact hf.integrableOn_isCompact hK
    /-
      🎉 no goals
    -/
    /-
      case hφ
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝² : NormedSpace Real E
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : T2Space X
      hf : MeasureTheory.LocallyIntegrable f μ
      g : X → Real
      hg : Continuous g
      h'g : HasCompactSupport g
      K : Set X := tsupport g
      hK : IsCompact K
      this : Eq (K.indicator fun x => HSMul.hSMul (g x) (f x)) fun x => HSMul.hSMul  …
      ⊢ MeasureTheory.Memℒp g Top.top μ
    -/
  · exact hg.memℒp_top_of_hasCompactSupport h'g μ
    /-
      🎉 no goals
    -/


/-- If `f` is locally integrable and `g` is continuous with compact support,
then `f • g` is integrable. -/
theorem LocallyIntegrable.integrable_smul_right_of_hasCompactSupport
    [NormedSpace ℝ E] [OpensMeasurableSpace X] [T2Space X] {f : X → ℝ} (hf : LocallyIntegrable f μ)
    {g : X → E} (hg : Continuous g) (h'g : HasCompactSupport g) :
    Integrable (fun x ↦ f x • g x) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    f : X → Real
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → E
    hg : Continuous g
    h'g : HasCompactSupport g
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (g x)) μ
  -/
  let K := tsupport g
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    f : X → Real
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → E
    hg : Continuous g
    h'g : HasCompactSupport g
    K : Set X := tsupport g
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (g x)) μ
  -/
  have hK : IsCompact K := h'g
  have : K.indicator (fun x ↦ f x • g x) = (fun x ↦ f x • g x) := by
    apply indicator_eq_self.2
    apply support_subset_iff'.2
    intros x hx
    simp [image_eq_zero_of_nmem_tsupport hx]
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    f : X → Real
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → E
    hg : Continuous g
    h'g : HasCompactSupport g
    K : Set X := tsupport g
    hK : IsCompact K
    this : Eq (K.indicator fun x => HSMul.hSMul (f x) (g x)) fun x => HSMul.hSMul  …
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (g x)) μ
  -/
  rw [← this, indicator_smul_left]
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝² : NormedSpace Real E
    inst✝¹ : OpensMeasurableSpace X
    inst✝ : T2Space X
    f : X → Real
    hf : MeasureTheory.LocallyIntegrable f μ
    g : X → E
    hg : Continuous g
    h'g : HasCompactSupport g
    K : Set X := tsupport g
    hK : IsCompact K
    this : Eq (K.indicator fun x => HSMul.hSMul (f x) (g x)) fun x => HSMul.hSMul  …
    ⊢ MeasureTheory.Integrable (fun a => HSMul.hSMul (K.indicator f a) (g a)) μ
  -/
  apply Integrable.smul_of_top_left
    /-
      case hφ
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝² : NormedSpace Real E
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : T2Space X
      f : X → Real
      hf : MeasureTheory.LocallyIntegrable f μ
      g : X → E
      hg : Continuous g
      h'g : HasCompactSupport g
      K : Set X := tsupport g
      hK : IsCompact K
      this : Eq (K.indicator fun x => HSMul.hSMul (f x) (g x)) fun x => HSMul.hSMul  …
      ⊢ MeasureTheory.Integrable (K.indicator f) μ
    -/
  · rw [integrable_indicator_iff hK.measurableSet]
    /-
      case hφ
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝² : NormedSpace Real E
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : T2Space X
      f : X → Real
      hf : MeasureTheory.LocallyIntegrable f μ
      g : X → E
      hg : Continuous g
      h'g : HasCompactSupport g
      K : Set X := tsupport g
      hK : IsCompact K
      this : Eq (K.indicator fun x => HSMul.hSMul (f x) (g x)) fun x => HSMul.hSMul  …
      ⊢ MeasureTheory.IntegrableOn f K μ
    -/
    exact hf.integrableOn_isCompact hK
    /-
      🎉 no goals
    -/
    /-
      case hf
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝² : NormedSpace Real E
      inst✝¹ : OpensMeasurableSpace X
      inst✝ : T2Space X
      f : X → Real
      hf : MeasureTheory.LocallyIntegrable f μ
      g : X → E
      hg : Continuous g
      h'g : HasCompactSupport g
      K : Set X := tsupport g
      hK : IsCompact K
      this : Eq (K.indicator fun x => HSMul.hSMul (f x) (g x)) fun x => HSMul.hSMul  …
      ⊢ MeasureTheory.Memℒp g Top.top μ
    -/
  · exact hg.memℒp_top_of_hasCompactSupport h'g μ
    /-
      🎉 no goals
    -/


theorem integrable_iff_integrableAtFilter_cocompact :
    Integrable f μ ↔ (IntegrableAtFilter f (cocompact X) μ ∧ LocallyIntegrable f μ) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    ⊢ Iff (MeasureTheory.Integrable f μ) (And (MeasureTheory.IntegrableAtFilter f  …
  -/
  refine ⟨fun hf ↦ ⟨hf.integrableAtFilter _, hf.locallyIntegrable⟩, fun ⟨⟨s, hsc, hs⟩, hloc⟩ ↦ ?_⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    x✝ : And (MeasureTheory.IntegrableAtFilter f (Filter.cocompact X) μ) (MeasureT …
    s : Set X
    hsc : Membership.mem (Filter.cocompact X) s
    hs : MeasureTheory.IntegrableOn f s μ
    hloc : MeasureTheory.LocallyIntegrable f μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  obtain ⟨t, htc, ht⟩ := mem_cocompact'.mp hsc
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    x✝ : And (MeasureTheory.IntegrableAtFilter f (Filter.cocompact X) μ) (MeasureT …
    s : Set X
    hsc : Membership.mem (Filter.cocompact X) s
    hs : MeasureTheory.IntegrableOn f s μ
    hloc : MeasureTheory.LocallyIntegrable f μ
    t : Set X
    htc : IsCompact t
    ht : HasSubset.Subset (HasCompl.compl s) t
    ⊢ MeasureTheory.Integrable f μ
  -/
  rewrite [← integrableOn_univ, ← compl_union_self s, integrableOn_union]
  /-
    case intro.intro
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : TopologicalSpace X
    inst✝ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    x✝ : And (MeasureTheory.IntegrableAtFilter f (Filter.cocompact X) μ) (MeasureT …
    s : Set X
    hsc : Membership.mem (Filter.cocompact X) s
    hs : MeasureTheory.IntegrableOn f s μ
    hloc : MeasureTheory.LocallyIntegrable f μ
    t : Set X
    htc : IsCompact t
    ht : HasSubset.Subset (HasCompl.compl s) t
    ⊢ And (MeasureTheory.IntegrableOn f (HasCompl.compl s) μ) (MeasureTheory.Integ …
  -/
  exact ⟨(hloc.integrableOn_isCompact htc).mono ht le_rfl, hs⟩
  /-
    🎉 no goals
  -/


theorem integrable_iff_integrableAtFilter_atBot_atTop [LinearOrder X] [CompactIccSpace X] :
    Integrable f μ ↔
    (IntegrableAtFilter f atBot μ ∧ IntegrableAtFilter f atTop μ) ∧ LocallyIntegrable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝¹ : LinearOrder X
    inst✝ : CompactIccSpace X
    ⊢ Iff (MeasureTheory.Integrable f μ) (And (And (MeasureTheory.IntegrableAtFilt …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      E : Type u_3
      inst✝⁴ : MeasurableSpace X
      inst✝³ : TopologicalSpace X
      inst✝² : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝¹ : LinearOrder X
      inst✝ : CompactIccSpace X
      ⊢ MeasureTheory.Integrable f μ → And (And (MeasureTheory.IntegrableAtFilter f  …
    -/
  · exact fun hf ↦ ⟨⟨hf.integrableAtFilter _, hf.integrableAtFilter _⟩, hf.locallyIntegrable⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      E : Type u_3
      inst✝⁴ : MeasurableSpace X
      inst✝³ : TopologicalSpace X
      inst✝² : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝¹ : LinearOrder X
      inst✝ : CompactIccSpace X
      ⊢ And (And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory. …
    -/
  · refine fun h ↦ integrable_iff_integrableAtFilter_cocompact.mpr ⟨?_, h.2⟩
    /-
      case mpr
      X : Type u_1
      E : Type u_3
      inst✝⁴ : MeasurableSpace X
      inst✝³ : TopologicalSpace X
      inst✝² : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝¹ : LinearOrder X
      inst✝ : CompactIccSpace X
      h : And (And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheor …
      ⊢ MeasureTheory.IntegrableAtFilter f (Filter.cocompact X) μ
    -/
    exact (IntegrableAtFilter.sup_iff.mpr h.1).filter_mono cocompact_le_atBot_atTop
    /-
      🎉 no goals
    -/


theorem integrable_iff_integrableAtFilter_atBot [LinearOrder X] [OrderTop X] [CompactIccSpace X] :
    Integrable f μ ↔ IntegrableAtFilter f atBot μ ∧ LocallyIntegrable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝² : LinearOrder X
    inst✝¹ : OrderTop X
    inst✝ : CompactIccSpace X
    ⊢ Iff (MeasureTheory.Integrable f μ) (And (MeasureTheory.IntegrableAtFilter f  …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝² : LinearOrder X
      inst✝¹ : OrderTop X
      inst✝ : CompactIccSpace X
      ⊢ MeasureTheory.Integrable f μ → And (MeasureTheory.IntegrableAtFilter f Filte …
    -/
  · exact fun hf ↦ ⟨hf.integrableAtFilter _, hf.locallyIntegrable⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝² : LinearOrder X
      inst✝¹ : OrderTop X
      inst✝ : CompactIccSpace X
      ⊢ And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory.Local …
    -/
  · refine fun h ↦ integrable_iff_integrableAtFilter_cocompact.mpr ⟨?_, h.2⟩
    /-
      case mpr
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝² : LinearOrder X
      inst✝¹ : OrderTop X
      inst✝ : CompactIccSpace X
      h : And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory.Loc …
      ⊢ MeasureTheory.IntegrableAtFilter f (Filter.cocompact X) μ
    -/
    exact h.1.filter_mono cocompact_le_atBot
    /-
      🎉 no goals
    -/


theorem integrable_iff_integrableAtFilter_atTop [LinearOrder X] [OrderBot X] [CompactIccSpace X] :
    Integrable f μ ↔ IntegrableAtFilter f atTop μ ∧ LocallyIntegrable f μ :=
  integrable_iff_integrableAtFilter_atBot (X := Xᵒᵈ)


theorem integrableOn_Iic_iff_integrableAtFilter_atBot [LinearOrder X] [CompactIccSpace X] :
    IntegrableOn f (Iic a) μ ↔ IntegrableAtFilter f atBot μ ∧ LocallyIntegrableOn f (Iic a) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    a : X
    inst✝¹ : LinearOrder X
    inst✝ : CompactIccSpace X
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Iic a) μ) (And (MeasureTheory.Integra …
  -/
  refine ⟨fun h ↦ ⟨⟨Iic a, Iic_mem_atBot a, h⟩, h.locallyIntegrableOn⟩, fun ⟨⟨s, hsl, hs⟩, h⟩ ↦ ?_⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    a : X
    inst✝¹ : LinearOrder X
    inst✝ : CompactIccSpace X
    x✝ : And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory.Lo …
    s : Set X
    hsl : Membership.mem Filter.atBot s
    hs : MeasureTheory.IntegrableOn f s μ
    h : MeasureTheory.LocallyIntegrableOn f (Set.Iic a) μ
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic a) μ
  -/
  haveI : Nonempty X := Nonempty.intro a
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    a : X
    inst✝¹ : LinearOrder X
    inst✝ : CompactIccSpace X
    x✝ : And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory.Lo …
    s : Set X
    hsl : Membership.mem Filter.atBot s
    hs : MeasureTheory.IntegrableOn f s μ
    h : MeasureTheory.LocallyIntegrableOn f (Set.Iic a) μ
    this : Nonempty X
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic a) μ
  -/
  obtain ⟨a', ha'⟩ := mem_atBot_sets.mp hsl
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    a : X
    inst✝¹ : LinearOrder X
    inst✝ : CompactIccSpace X
    x✝ : And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory.Lo …
    s : Set X
    hsl : Membership.mem Filter.atBot s
    hs : MeasureTheory.IntegrableOn f s μ
    h : MeasureTheory.LocallyIntegrableOn f (Set.Iic a) μ
    this : Nonempty X
    a' : X
    ha' : ∀ (b : X), LE.le b a' → Membership.mem s b
    ⊢ MeasureTheory.IntegrableOn f (Set.Iic a) μ
  -/
  refine (integrableOn_union.mpr ⟨hs.mono ha' le_rfl, ?_⟩).mono Iic_subset_Iic_union_Icc le_rfl
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    a : X
    inst✝¹ : LinearOrder X
    inst✝ : CompactIccSpace X
    x✝ : And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (MeasureTheory.Lo …
    s : Set X
    hsl : Membership.mem Filter.atBot s
    hs : MeasureTheory.IntegrableOn f s μ
    h : MeasureTheory.LocallyIntegrableOn f (Set.Iic a) μ
    this : Nonempty X
    a' : X
    ha' : ∀ (b : X), LE.le b a' → Membership.mem s b
    ⊢ MeasureTheory.IntegrableOn f (Set.Icc a' a) μ
  -/
  exact h.integrableOn_compact_subset Icc_subset_Iic_self isCompact_Icc
  /-
    🎉 no goals
  -/


theorem integrableOn_Ici_iff_integrableAtFilter_atTop [LinearOrder X] [CompactIccSpace X] :
    IntegrableOn f (Ici a) μ ↔ IntegrableAtFilter f atTop μ ∧ LocallyIntegrableOn f (Ici a) μ :=
  integrableOn_Iic_iff_integrableAtFilter_atBot (X := Xᵒᵈ)


theorem integrableOn_Iio_iff_integrableAtFilter_atBot_nhdsWithin
    [LinearOrder X] [CompactIccSpace X] [NoMinOrder X] [OrderTopology X] :
    IntegrableOn f (Iio a) μ ↔ IntegrableAtFilter f atBot μ ∧
    IntegrableAtFilter f (𝓝[<] a) μ ∧ LocallyIntegrableOn f (Iio a) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    a : X
    inst✝³ : LinearOrder X
    inst✝² : CompactIccSpace X
    inst✝¹ : NoMinOrder X
    inst✝ : OrderTopology X
    ⊢ Iff (MeasureTheory.IntegrableOn f (Set.Iio a) μ) (And (MeasureTheory.Integra …
  -/
  constructor
    /-
      case mp
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      a : X
      inst✝³ : LinearOrder X
      inst✝² : CompactIccSpace X
      inst✝¹ : NoMinOrder X
      inst✝ : OrderTopology X
      ⊢ MeasureTheory.IntegrableOn f (Set.Iio a) μ → And (MeasureTheory.IntegrableAt …
    -/
  · intro h
    /-
      case mp
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      a : X
      inst✝³ : LinearOrder X
      inst✝² : CompactIccSpace X
      inst✝¹ : NoMinOrder X
      inst✝ : OrderTopology X
      h : MeasureTheory.IntegrableOn f (Set.Iio a) μ
      ⊢ And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (And (MeasureTheory. …
    -/
    exact ⟨⟨Iio a, Iio_mem_atBot a, h⟩, ⟨Iio a, self_mem_nhdsWithin, h⟩, h.locallyIntegrableOn⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      a : X
      inst✝³ : LinearOrder X
      inst✝² : CompactIccSpace X
      inst✝¹ : NoMinOrder X
      inst✝ : OrderTopology X
      ⊢ And (MeasureTheory.IntegrableAtFilter f Filter.atBot μ) (And (MeasureTheory. …
    -/
  · intro ⟨hbot, ⟨s, hsl, hs⟩, hlocal⟩
    /-
      case mpr
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      a : X
      inst✝³ : LinearOrder X
      inst✝² : CompactIccSpace X
      inst✝¹ : NoMinOrder X
      inst✝ : OrderTopology X
      hbot : MeasureTheory.IntegrableAtFilter f Filter.atBot μ
      s : Set X
      hsl : Membership.mem (nhdsWithin a (Set.Iio a)) s
      hs : MeasureTheory.IntegrableOn f s μ
      hlocal : MeasureTheory.LocallyIntegrableOn f (Set.Iio a) μ
      ⊢ MeasureTheory.IntegrableOn f (Set.Iio a) μ
    -/
    obtain ⟨s', ⟨hs'_mono, hs'⟩⟩ := mem_nhdsLT_iff_exists_Ioo_subset.mp hsl
    /-
      case mpr.intro.intro
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      a : X
      inst✝³ : LinearOrder X
      inst✝² : CompactIccSpace X
      inst✝¹ : NoMinOrder X
      inst✝ : OrderTopology X
      hbot : MeasureTheory.IntegrableAtFilter f Filter.atBot μ
      s : Set X
      hsl : Membership.mem (nhdsWithin a (Set.Iio a)) s
      hs : MeasureTheory.IntegrableOn f s μ
      hlocal : MeasureTheory.LocallyIntegrableOn f (Set.Iio a) μ
      s' : X
      hs'_mono : Membership.mem (Set.Iio a) s'
      hs' : HasSubset.Subset (Set.Ioo s' a) s
      ⊢ MeasureTheory.IntegrableOn f (Set.Iio a) μ
    -/
    refine (integrableOn_union.mpr ⟨?_, hs.mono hs' le_rfl⟩).mono Iio_subset_Iic_union_Ioo le_rfl
    exact integrableOn_Iic_iff_integrableAtFilter_atBot.mpr
      ⟨hbot, hlocal.mono_set (Iic_subset_Iio.mpr hs'_mono)⟩


theorem integrableOn_Ioi_iff_integrableAtFilter_atTop_nhdsWithin
    [LinearOrder X] [CompactIccSpace X] [NoMaxOrder X] [OrderTopology X] :
    IntegrableOn f (Ioi a) μ ↔ IntegrableAtFilter f atTop μ ∧
    IntegrableAtFilter f (𝓝[>] a) μ ∧ LocallyIntegrableOn f (Ioi a) μ :=
  integrableOn_Iio_iff_integrableAtFilter_atBot_nhdsWithin (X := Xᵒᵈ)


/-- A continuous function `f` is locally integrable with respect to any locally finite measure. -/
theorem Continuous.locallyIntegrable [IsLocallyFiniteMeasure μ] [SecondCountableTopologyEither X E]
    (hf : Continuous f) : LocallyIntegrable f μ :=
  hf.integrableAt_nhds


/-- A function `f` continuous on a set `K` is locally integrable on this set with respect
to any locally finite measure. -/
theorem ContinuousOn.locallyIntegrableOn [IsLocallyFiniteMeasure μ]
    [SecondCountableTopologyEither X E] (hf : ContinuousOn f K)
    (hK : MeasurableSet K) : LocallyIntegrableOn f K μ := fun _x hx =>
  hf.integrableAt_nhdsWithin hK hx


/-- A function `f` continuous on a compact set `K` is integrable on this set with respect to any
locally finite measure. -/
theorem ContinuousOn.integrableOn_compact'
    (hK : IsCompact K) (h'K : MeasurableSet K) (hf : ContinuousOn f K) :
    IntegrableOn f K μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝¹ : OpensMeasurableSpace X
    K : Set X
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    hK : IsCompact K
    h'K : MeasurableSet K
    hf : ContinuousOn f K
    ⊢ MeasureTheory.IntegrableOn f K μ
  -/
  refine ⟨ContinuousOn.aestronglyMeasurable_of_isCompact hf hK h'K, ?_⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝¹ : OpensMeasurableSpace X
    K : Set X
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    hK : IsCompact K
    h'K : MeasurableSet K
    hf : ContinuousOn f K
    ⊢ MeasureTheory.HasFiniteIntegral f (μ.restrict K)
  -/
  have : Fact (μ K < ∞) := ⟨hK.measure_lt_top⟩
  obtain ⟨C, hC⟩ : ∃ C, ∀ x ∈ f '' K, ‖x‖ ≤ C :=
    IsBounded.exists_norm_le (hK.image_of_continuousOn hf).isBounded
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝¹ : OpensMeasurableSpace X
    K : Set X
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    hK : IsCompact K
    h'K : MeasurableSet K
    hf : ContinuousOn f K
    this : Fact (LT.lt (μ K) Top.top)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f K) x → LE.le (Norm.norm x) C
    ⊢ MeasureTheory.HasFiniteIntegral f (μ.restrict K)
  -/
  apply hasFiniteIntegral_of_bounded (C := C)
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    inst✝² : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝¹ : OpensMeasurableSpace X
    K : Set X
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    hK : IsCompact K
    h'K : MeasurableSet K
    hf : ContinuousOn f K
    this : Fact (LT.lt (μ K) Top.top)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f K) x → LE.le (Norm.norm x) C
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f a)) C) (MeasureTheory.ae (μ. …
  -/
  filter_upwards [ae_restrict_mem h'K] with x hx using hC _ (mem_image_of_mem f hx)
  /-
    🎉 no goals
  -/


theorem ContinuousOn.integrableOn_compact [T2Space X]
    (hK : IsCompact K) (hf : ContinuousOn f K) : IntegrableOn f K μ :=
  hf.integrableOn_compact' hK hK.measurableSet


theorem ContinuousOn.integrableOn_Icc [Preorder X] [CompactIccSpace X] [T2Space X]
    (hf : ContinuousOn f (Icc a b)) : IntegrableOn f (Icc a b) μ :=
  hf.integrableOn_compact isCompact_Icc


theorem Continuous.integrableOn_Icc [Preorder X] [CompactIccSpace X] [T2Space X]
    (hf : Continuous f) : IntegrableOn f (Icc a b) μ :=
  hf.continuousOn.integrableOn_Icc


theorem Continuous.integrableOn_Ioc [Preorder X] [CompactIccSpace X] [T2Space X]
    (hf : Continuous f) : IntegrableOn f (Ioc a b) μ :=
  hf.integrableOn_Icc.mono_set Ioc_subset_Icc_self


theorem ContinuousOn.integrableOn_uIcc [LinearOrder X] [CompactIccSpace X] [T2Space X]
    (hf : ContinuousOn f [[a, b]]) : IntegrableOn f [[a, b]] μ :=
  hf.integrableOn_Icc


theorem Continuous.integrableOn_uIcc [LinearOrder X] [CompactIccSpace X] [T2Space X]
    (hf : Continuous f) : IntegrableOn f [[a, b]] μ :=
  hf.integrableOn_Icc


theorem Continuous.integrableOn_uIoc [LinearOrder X] [CompactIccSpace X] [T2Space X]
    (hf : Continuous f) : IntegrableOn f (Ι a b) μ :=
  hf.integrableOn_Ioc


/-- A continuous function with compact support is integrable on the whole space. -/
theorem Continuous.integrable_of_hasCompactSupport (hf : Continuous f) (hcf : HasCompactSupport f) :
    Integrable f μ :=
  (integrableOn_iff_integrable_of_support_subset (subset_tsupport f)).mp <|
    hf.continuousOn.integrableOn_compact' hcf (isClosed_tsupport _).measurableSet


theorem MonotoneOn.memℒp_top (hmono : MonotoneOn f s) {a b : X}
    (ha : IsLeast s a) (hb : IsGreatest s b) (h's : MeasurableSet s) :
    Memℒp f ∞ (μ.restrict s) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  borelize E
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  have hbelow : BddBelow (f '' s) := ⟨f a, fun x ⟨y, hy, hyx⟩ => hyx ▸ hmono ha.1 hy (ha.2 hy)⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  have habove : BddAbove (f '' s) := ⟨f b, fun x ⟨y, hy, hyx⟩ => hyx ▸ hmono hy hb.1 (hb.2 hy)⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  have : IsBounded (f '' s) := Metric.isBounded_of_bddAbove_of_bddBelow habove hbelow
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    this : Bornology.IsBounded (Set.image f s)
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  rcases isBounded_iff_forall_norm_le.mp this with ⟨C, hC⟩
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    this : Bornology.IsBounded (Set.image f s)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f s) x → LE.le (Norm.norm x) C
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  have A : Memℒp (fun _ => C) ⊤ (μ.restrict s) := memℒp_top_const _
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    this : Bornology.IsBounded (Set.image f s)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f s) x → LE.le (Norm.norm x) C
    A : MeasureTheory.Memℒp (fun x => C) Top.top (μ.restrict s)
    ⊢ MeasureTheory.Memℒp f Top.top (μ.restrict s)
  -/
  apply Memℒp.mono A (aemeasurable_restrict_of_monotoneOn h's hmono).aestronglyMeasurable
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    this : Bornology.IsBounded (Set.image f s)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f s) x → LE.le (Norm.norm x) C
    A : MeasureTheory.Memℒp (fun x => C) Top.top (μ.restrict s)
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Norm.norm C)) (MeasureT …
  -/
  apply (ae_restrict_iff' h's).mpr
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    this : Bornology.IsBounded (Set.image f s)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f s) x → LE.le (Norm.norm x) C
    A : MeasureTheory.Memℒp (fun x => C) Top.top (μ.restrict s)
    ⊢ Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (f x)) (No …
  -/
  apply ae_of_all _ fun y hy ↦ ?_
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : ConditionallyCompleteLinearOrder X
    inst✝³ : ConditionallyCompleteLinearOrder E
    inst✝² : OrderTopology X
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    hmono : MonotoneOn f s
    a b : X
    ha : IsLeast s a
    hb : IsGreatest s b
    h's : MeasurableSet s
    this✝¹ : MeasurableSpace E := borel E
    this✝ : BorelSpace E
    hbelow : BddBelow (Set.image f s)
    habove : BddAbove (Set.image f s)
    this : Bornology.IsBounded (Set.image f s)
    C : Real
    hC : ∀ (x : E), Membership.mem (Set.image f s) x → LE.le (Norm.norm x) C
    A : MeasureTheory.Memℒp (fun x => C) Top.top (μ.restrict s)
    y : X
    hy : Membership.mem s y
    ⊢ LE.le (Norm.norm (f y)) (Norm.norm C)
  -/
  exact (hC _ (mem_image_of_mem f hy)).trans (le_abs_self _)
  /-
    🎉 no goals
  -/


theorem MonotoneOn.memℒp_of_measure_ne_top (hmono : MonotoneOn f s) {a b : X}
    (ha : IsLeast s a) (hb : IsGreatest s b) (hs : μ s ≠ ∞) (h's : MeasurableSet s) :
    Memℒp f p (μ.restrict s) :=
  (hmono.memℒp_top ha hb h's).memℒp_of_exponent_le_of_measure_support_ne_top (s := univ)
        /-
          X : Type u_1
          E : Type u_3
          inst✝⁸ : MeasurableSpace X
          inst✝⁷ : TopologicalSpace X
          inst✝⁶ : NormedAddCommGroup E
          f : X → E
          μ : MeasureTheory.Measure X
          s : Set X
          inst✝⁵ : BorelSpace X
          inst✝⁴ : ConditionallyCompleteLinearOrder X
          inst✝³ : ConditionallyCompleteLinearOrder E
          inst✝² : OrderTopology X
          inst✝¹ : OrderTopology E
          inst✝ : SecondCountableTopology E
          p : ENNReal
          hmono : MonotoneOn f s
          a b : X
          ha : IsLeast s a
          hb : IsGreatest s b
          hs : Ne (μ s) Top.top
          h's : MeasurableSet s
          ⊢ ∀ (x : X), Not (Membership.mem Set.univ x) → Eq (f x) 0
        -/
        /-
          🎉 no goals
        -/
    (by simp) (by simpa using hs) le_top
                  /-
                    🎉 no goals
                  -/


theorem MonotoneOn.memℒp_isCompact [IsFiniteMeasureOnCompacts μ] (hs : IsCompact s)
    (hmono : MonotoneOn f s) : Memℒp f p (μ.restrict s) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    s : Set X
    inst✝⁶ : BorelSpace X
    inst✝⁵ : ConditionallyCompleteLinearOrder X
    inst✝⁴ : ConditionallyCompleteLinearOrder E
    inst✝³ : OrderTopology X
    inst✝² : OrderTopology E
    inst✝¹ : SecondCountableTopology E
    p : ENNReal
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    hs : IsCompact s
    hmono : MonotoneOn f s
    ⊢ MeasureTheory.Memℒp f p (μ.restrict s)
  -/
  obtain rfl | h := s.eq_empty_or_nonempty
    /-
      case inl
      X : Type u_1
      E : Type u_3
      inst✝⁹ : MeasurableSpace X
      inst✝⁸ : TopologicalSpace X
      inst✝⁷ : NormedAddCommGroup E
      f : X → E
      μ : MeasureTheory.Measure X
      inst✝⁶ : BorelSpace X
      inst✝⁵ : ConditionallyCompleteLinearOrder X
      inst✝⁴ : ConditionallyCompleteLinearOrder E
      inst✝³ : OrderTopology X
      inst✝² : OrderTopology E
      inst✝¹ : SecondCountableTopology E
      p : ENNReal
      inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      hs : IsCompact EmptyCollection.emptyCollection
      hmono : MonotoneOn f EmptyCollection.emptyCollection
      ⊢ MeasureTheory.Memℒp f p (μ.restrict EmptyCollection.emptyCollection)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · exact hmono.memℒp_of_measure_ne_top (hs.isLeast_sInf h) (hs.isGreatest_sSup h)
      hs.measure_lt_top.ne hs.measurableSet


theorem AntitoneOn.memℒp_top (hanti : AntitoneOn f s) {a b : X}
    (ha : IsLeast s a) (hb : IsGreatest s b) (h's : MeasurableSet s) :
    Memℒp f ∞ (μ.restrict s) :=
  MonotoneOn.memℒp_top (E := Eᵒᵈ) hanti ha hb h's


theorem AntitoneOn.memℒp_of_measure_ne_top (hanti : AntitoneOn f s) {a b : X}
    (ha : IsLeast s a) (hb : IsGreatest s b) (hs : μ s ≠ ∞) (h's : MeasurableSet s) :
    Memℒp f p (μ.restrict s) :=
  MonotoneOn.memℒp_of_measure_ne_top (E := Eᵒᵈ) hanti ha hb hs h's


theorem AntitoneOn.memℒp_isCompact [IsFiniteMeasureOnCompacts μ] (hs : IsCompact s)
    (hanti : AntitoneOn f s) : Memℒp f p (μ.restrict s) :=
  MonotoneOn.memℒp_isCompact (E := Eᵒᵈ) hs hanti


theorem MonotoneOn.integrableOn_of_measure_ne_top (hmono : MonotoneOn f s) {a b : X}
    (ha : IsLeast s a) (hb : IsGreatest s b) (hs : μ s ≠ ∞) (h's : MeasurableSet s) :
    IntegrableOn f s μ :=
  memℒp_one_iff_integrable.1 (hmono.memℒp_of_measure_ne_top ha hb hs h's)


theorem MonotoneOn.integrableOn_isCompact [IsFiniteMeasureOnCompacts μ] (hs : IsCompact s)
    (hmono : MonotoneOn f s) : IntegrableOn f s μ :=
  memℒp_one_iff_integrable.1 (hmono.memℒp_isCompact hs)


theorem AntitoneOn.integrableOn_of_measure_ne_top (hanti : AntitoneOn f s) {a b : X}
    (ha : IsLeast s a) (hb : IsGreatest s b) (hs : μ s ≠ ∞) (h's : MeasurableSet s) :
    IntegrableOn f s μ :=
  memℒp_one_iff_integrable.1 (hanti.memℒp_of_measure_ne_top ha hb hs h's)


theorem AntioneOn.integrableOn_isCompact [IsFiniteMeasureOnCompacts μ] (hs : IsCompact s)
    (hanti : AntitoneOn f s) : IntegrableOn f s μ :=
  memℒp_one_iff_integrable.1 (hanti.memℒp_isCompact hs)


theorem Monotone.locallyIntegrable [IsLocallyFiniteMeasure μ] (hmono : Monotone f) :
    LocallyIntegrable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝⁶ : BorelSpace X
    inst✝⁵ : ConditionallyCompleteLinearOrder X
    inst✝⁴ : ConditionallyCompleteLinearOrder E
    inst✝³ : OrderTopology X
    inst✝² : OrderTopology E
    inst✝¹ : SecondCountableTopology E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hmono : Monotone f
    ⊢ MeasureTheory.LocallyIntegrable f μ
  -/
  intro x
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝⁶ : BorelSpace X
    inst✝⁵ : ConditionallyCompleteLinearOrder X
    inst✝⁴ : ConditionallyCompleteLinearOrder E
    inst✝³ : OrderTopology X
    inst✝² : OrderTopology E
    inst✝¹ : SecondCountableTopology E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hmono : Monotone f
    x : X
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) μ
  -/
  rcases μ.finiteAt_nhds x with ⟨U, hU, h'U⟩
  obtain ⟨a, b, xab, hab, abU⟩ : ∃ a b : X, x ∈ Icc a b ∧ Icc a b ∈ 𝓝 x ∧ Icc a b ⊆ U :=
    exists_Icc_mem_subset_of_mem_nhds hU
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝⁶ : BorelSpace X
    inst✝⁵ : ConditionallyCompleteLinearOrder X
    inst✝⁴ : ConditionallyCompleteLinearOrder E
    inst✝³ : OrderTopology X
    inst✝² : OrderTopology E
    inst✝¹ : SecondCountableTopology E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hmono : Monotone f
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : LT.lt (μ U) Top.top
    a b : X
    xab : Membership.mem (Set.Icc a b) x
    hab : Membership.mem (nhds x) (Set.Icc a b)
    abU : HasSubset.Subset (Set.Icc a b) U
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) μ
  -/
  have ab : a ≤ b := xab.1.trans xab.2
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    E : Type u_3
    inst✝⁹ : MeasurableSpace X
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : NormedAddCommGroup E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝⁶ : BorelSpace X
    inst✝⁵ : ConditionallyCompleteLinearOrder X
    inst✝⁴ : ConditionallyCompleteLinearOrder E
    inst✝³ : OrderTopology X
    inst✝² : OrderTopology E
    inst✝¹ : SecondCountableTopology E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    hmono : Monotone f
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    h'U : LT.lt (μ U) Top.top
    a b : X
    xab : Membership.mem (Set.Icc a b) x
    hab : Membership.mem (nhds x) (Set.Icc a b)
    abU : HasSubset.Subset (Set.Icc a b) U
    ab : LE.le a b
    ⊢ MeasureTheory.IntegrableAtFilter f (nhds x) μ
  -/
  refine ⟨Icc a b, hab, ?_⟩
  exact
    (hmono.monotoneOn _).integrableOn_of_measure_ne_top (isLeast_Icc ab) (isGreatest_Icc ab)
      ((measure_mono abU).trans_lt h'U).ne measurableSet_Icc


theorem Antitone.locallyIntegrable [IsLocallyFiniteMeasure μ] (hanti : Antitone f) :
    LocallyIntegrable f μ :=
  hanti.dual_right.locallyIntegrable


theorem IntegrableOn.mul_continuousOn_of_subset (hg : IntegrableOn g A μ) (hg' : ContinuousOn g' K)
    (hA : MeasurableSet A) (hK : IsCompact K) (hAK : A ⊆ K) :
    IntegrableOn (fun x => g x * g' x) A μ := by
  /-
    X : Type u_1
    R : Type u_5
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝² : OpensMeasurableSpace X
    A K : Set X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    g g' : X → R
    hg : MeasureTheory.IntegrableOn g A μ
    hg' : ContinuousOn g' K
    hA : MeasurableSet A
    hK : IsCompact K
    hAK : HasSubset.Subset A K
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (g x) (g' x)) A μ
  -/
  rcases IsCompact.exists_bound_of_continuousOn hK hg' with ⟨C, hC⟩
  /-
    case intro
    X : Type u_1
    R : Type u_5
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝² : OpensMeasurableSpace X
    A K : Set X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    g g' : X → R
    hg : MeasureTheory.IntegrableOn g A μ
    hg' : ContinuousOn g' K
    hA : MeasurableSet A
    hK : IsCompact K
    hAK : HasSubset.Subset A K
    C : Real
    hC : ∀ (x : X), Membership.mem K x → LE.le (Norm.norm (g' x)) C
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (g x) (g' x)) A μ
  -/
  rw [IntegrableOn, ← memℒp_one_iff_integrable] at hg ⊢
  have : ∀ᵐ x ∂μ.restrict A, ‖g x * g' x‖ ≤ C * ‖g x‖ := by
    filter_upwards [ae_restrict_mem hA] with x hx
    refine (norm_mul_le _ _).trans ?_
    rw [mul_comm]
    gcongr
    exact hC x (hAK hx)
  exact
    Memℒp.of_le_mul hg (hg.aestronglyMeasurable.mul <| (hg'.mono hAK).aestronglyMeasurable hA) this


theorem IntegrableOn.mul_continuousOn [T2Space X] (hg : IntegrableOn g K μ)
    (hg' : ContinuousOn g' K) (hK : IsCompact K) : IntegrableOn (fun x => g x * g' x) K μ :=
  hg.mul_continuousOn_of_subset hg' hK.measurableSet hK (Subset.refl _)


theorem IntegrableOn.continuousOn_mul_of_subset (hg : ContinuousOn g K) (hg' : IntegrableOn g' A μ)
    (hK : IsCompact K) (hA : MeasurableSet A) (hAK : A ⊆ K) :
    IntegrableOn (fun x => g x * g' x) A μ := by
  /-
    X : Type u_1
    R : Type u_5
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝² : OpensMeasurableSpace X
    A K : Set X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    g g' : X → R
    hg : ContinuousOn g K
    hg' : MeasureTheory.IntegrableOn g' A μ
    hK : IsCompact K
    hA : MeasurableSet A
    hAK : HasSubset.Subset A K
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (g x) (g' x)) A μ
  -/
  rcases IsCompact.exists_bound_of_continuousOn hK hg with ⟨C, hC⟩
  /-
    case intro
    X : Type u_1
    R : Type u_5
    inst✝⁴ : MeasurableSpace X
    inst✝³ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝² : OpensMeasurableSpace X
    A K : Set X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    g g' : X → R
    hg : ContinuousOn g K
    hg' : MeasureTheory.IntegrableOn g' A μ
    hK : IsCompact K
    hA : MeasurableSet A
    hAK : HasSubset.Subset A K
    C : Real
    hC : ∀ (x : X), Membership.mem K x → LE.le (Norm.norm (g x)) C
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (g x) (g' x)) A μ
  -/
  rw [IntegrableOn, ← memℒp_one_iff_integrable] at hg' ⊢
  have : ∀ᵐ x ∂μ.restrict A, ‖g x * g' x‖ ≤ C * ‖g' x‖ := by
    filter_upwards [ae_restrict_mem hA] with x hx
    refine (norm_mul_le _ _).trans ?_
    gcongr
    exact hC x (hAK hx)
  exact
    Memℒp.of_le_mul hg' (((hg.mono hAK).aestronglyMeasurable hA).mul hg'.aestronglyMeasurable) this


theorem IntegrableOn.continuousOn_mul [T2Space X] (hg : ContinuousOn g K)
    (hg' : IntegrableOn g' K μ) (hK : IsCompact K) : IntegrableOn (fun x => g x * g' x) K μ :=
  hg'.continuousOn_mul_of_subset hg hK hK.measurableSet Subset.rfl


theorem IntegrableOn.continuousOn_smul [T2Space X] [SecondCountableTopologyEither X 𝕜] {g : X → E}
    (hg : IntegrableOn g K μ) {f : X → 𝕜} (hf : ContinuousOn f K) (hK : IsCompact K) :
    IntegrableOn (fun x => f x • g x) K μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁷ : MeasurableSpace X
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝⁴ : OpensMeasurableSpace X
    K : Set X
    𝕜 : Type u_6
    inst✝³ : NormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : T2Space X
    inst✝ : SecondCountableTopologyEither X 𝕜
    g : X → E
    hg : MeasureTheory.IntegrableOn g K μ
    f : X → 𝕜
    hf : ContinuousOn f K
    hK : IsCompact K
    ⊢ MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f x) (g x)) K μ
  -/
  rw [IntegrableOn, ← integrable_norm_iff]
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X 𝕜
      g : X → E
      hg : MeasureTheory.IntegrableOn g K μ
      f : X → 𝕜
      hf : ContinuousOn f K
      hK : IsCompact K
      ⊢ MeasureTheory.Integrable (fun a => Norm.norm (HSMul.hSMul (f a) (g a))) (μ.r …
    -/
  · simp_rw [norm_smul]
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X 𝕜
      g : X → E
      hg : MeasureTheory.IntegrableOn g K μ
      f : X → 𝕜
      hf : ContinuousOn f K
      hK : IsCompact K
      ⊢ MeasureTheory.Integrable (fun a => HMul.hMul (Norm.norm (f a)) (Norm.norm (g …
    -/
    refine IntegrableOn.continuousOn_mul ?_ hg.norm hK
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X 𝕜
      g : X → E
      hg : MeasureTheory.IntegrableOn g K μ
      f : X → 𝕜
      hf : ContinuousOn f K
      hK : IsCompact K
      ⊢ ContinuousOn (fun a => Norm.norm (f a)) K
    -/
    exact continuous_norm.comp_continuousOn hf
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X 𝕜
      g : X → E
      hg : MeasureTheory.IntegrableOn g K μ
      f : X → 𝕜
      hf : ContinuousOn f K
      hK : IsCompact K
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (f x) (g x)) (μ.res …
    -/
  · exact (hf.aestronglyMeasurable hK.measurableSet).smul hg.1
    /-
      🎉 no goals
    -/


theorem IntegrableOn.smul_continuousOn [T2Space X] [SecondCountableTopologyEither X E] {f : X → 𝕜}
    (hf : IntegrableOn f K μ) {g : X → E} (hg : ContinuousOn g K) (hK : IsCompact K) :
    IntegrableOn (fun x => f x • g x) K μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁷ : MeasurableSpace X
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝⁴ : OpensMeasurableSpace X
    K : Set X
    𝕜 : Type u_6
    inst✝³ : NormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : T2Space X
    inst✝ : SecondCountableTopologyEither X E
    f : X → 𝕜
    hf : MeasureTheory.IntegrableOn f K μ
    g : X → E
    hg : ContinuousOn g K
    hK : IsCompact K
    ⊢ MeasureTheory.IntegrableOn (fun x => HSMul.hSMul (f x) (g x)) K μ
  -/
  rw [IntegrableOn, ← integrable_norm_iff]
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X E
      f : X → 𝕜
      hf : MeasureTheory.IntegrableOn f K μ
      g : X → E
      hg : ContinuousOn g K
      hK : IsCompact K
      ⊢ MeasureTheory.Integrable (fun a => Norm.norm (HSMul.hSMul (f a) (g a))) (μ.r …
    -/
  · simp_rw [norm_smul]
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X E
      f : X → 𝕜
      hf : MeasureTheory.IntegrableOn f K μ
      g : X → E
      hg : ContinuousOn g K
      hK : IsCompact K
      ⊢ MeasureTheory.Integrable (fun a => HMul.hMul (Norm.norm (f a)) (Norm.norm (g …
    -/
    refine IntegrableOn.mul_continuousOn hf.norm ?_ hK
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X E
      f : X → 𝕜
      hf : MeasureTheory.IntegrableOn f K μ
      g : X → E
      hg : ContinuousOn g K
      hK : IsCompact K
      ⊢ ContinuousOn (fun a => Norm.norm (g a)) K
    -/
    exact continuous_norm.comp_continuousOn hg
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : NormedAddCommGroup E
      μ : MeasureTheory.Measure X
      inst✝⁴ : OpensMeasurableSpace X
      K : Set X
      𝕜 : Type u_6
      inst✝³ : NormedField 𝕜
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : T2Space X
      inst✝ : SecondCountableTopologyEither X E
      f : X → 𝕜
      hf : MeasureTheory.IntegrableOn f K μ
      g : X → E
      hg : ContinuousOn g K
      hK : IsCompact K
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (f x) (g x)) (μ.res …
    -/
  · exact hf.1.smul (hg.aestronglyMeasurable hK.measurableSet)
    /-
      🎉 no goals
    -/


theorem continuousOn_mul [LocallyCompactSpace X] [T2Space X] [NormedRing R]
    [SecondCountableTopologyEither X R] {f g : X → R} {s : Set X} (hf : LocallyIntegrableOn f s μ)
    (hg : ContinuousOn g s) (hs : IsLocallyClosed s) :
    LocallyIntegrableOn (fun x => g x * f x) s μ := by
  /-
    X : Type u_1
    R : Type u_5
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : OpensMeasurableSpace X
    inst✝³ : LocallyCompactSpace X
    inst✝² : T2Space X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    f g : X → R
    s : Set X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    hg : ContinuousOn g s
    hs : IsLocallyClosed s
    ⊢ MeasureTheory.LocallyIntegrableOn (fun x => HMul.hMul (g x) (f x)) s μ
  -/
  rw [MeasureTheory.locallyIntegrableOn_iff hs] at hf ⊢
  /-
    X : Type u_1
    R : Type u_5
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : OpensMeasurableSpace X
    inst✝³ : LocallyCompactSpace X
    inst✝² : T2Space X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    f g : X → R
    s : Set X
    hf : ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integra …
    hg : ContinuousOn g s
    hs : IsLocallyClosed s
    ⊢ ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integrable …
  -/
  exact fun k hk_sub hk_c => (hf k hk_sub hk_c).continuousOn_mul (hg.mono hk_sub) hk_c
  /-
    🎉 no goals
  -/


theorem mul_continuousOn [LocallyCompactSpace X] [T2Space X] [NormedRing R]
    [SecondCountableTopologyEither X R] {f g : X → R} {s : Set X} (hf : LocallyIntegrableOn f s μ)
    (hg : ContinuousOn g s) (hs : IsLocallyClosed s) :
    LocallyIntegrableOn (fun x => f x * g x) s μ := by
  /-
    X : Type u_1
    R : Type u_5
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : OpensMeasurableSpace X
    inst✝³ : LocallyCompactSpace X
    inst✝² : T2Space X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    f g : X → R
    s : Set X
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    hg : ContinuousOn g s
    hs : IsLocallyClosed s
    ⊢ MeasureTheory.LocallyIntegrableOn (fun x => HMul.hMul (f x) (g x)) s μ
  -/
  rw [MeasureTheory.locallyIntegrableOn_iff hs] at hf ⊢
  /-
    X : Type u_1
    R : Type u_5
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : TopologicalSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : OpensMeasurableSpace X
    inst✝³ : LocallyCompactSpace X
    inst✝² : T2Space X
    inst✝¹ : NormedRing R
    inst✝ : SecondCountableTopologyEither X R
    f g : X → R
    s : Set X
    hf : ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integra …
    hg : ContinuousOn g s
    hs : IsLocallyClosed s
    ⊢ ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integrable …
  -/
  exact fun k hk_sub hk_c => (hf k hk_sub hk_c).mul_continuousOn (hg.mono hk_sub) hk_c
  /-
    🎉 no goals
  -/


theorem continuousOn_smul [LocallyCompactSpace X] [T2Space X] {𝕜 : Type*} [NormedField 𝕜]
    [SecondCountableTopologyEither X 𝕜] [NormedSpace 𝕜 E] {f : X → E} {g : X → 𝕜} {s : Set X}
    (hs : IsLocallyClosed s) (hf : LocallyIntegrableOn f s μ) (hg : ContinuousOn g s) :
    LocallyIntegrableOn (fun x => g x • f x) s μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : LocallyCompactSpace X
    inst✝³ : T2Space X
    𝕜 : Type u_6
    inst✝² : NormedField 𝕜
    inst✝¹ : SecondCountableTopologyEither X 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : X → E
    g : X → 𝕜
    s : Set X
    hs : IsLocallyClosed s
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    hg : ContinuousOn g s
    ⊢ MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (g x) (f x)) s μ
  -/
  rw [MeasureTheory.locallyIntegrableOn_iff hs] at hf ⊢
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : LocallyCompactSpace X
    inst✝³ : T2Space X
    𝕜 : Type u_6
    inst✝² : NormedField 𝕜
    inst✝¹ : SecondCountableTopologyEither X 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : X → E
    g : X → 𝕜
    s : Set X
    hs : IsLocallyClosed s
    hf : ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integra …
    hg : ContinuousOn g s
    ⊢ ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integrable …
  -/
  exact fun k hk_sub hk_c => (hf k hk_sub hk_c).continuousOn_smul (hg.mono hk_sub) hk_c
  /-
    🎉 no goals
  -/


theorem smul_continuousOn [LocallyCompactSpace X] [T2Space X] {𝕜 : Type*} [NormedField 𝕜]
    [SecondCountableTopologyEither X E] [NormedSpace 𝕜 E] {f : X → 𝕜} {g : X → E} {s : Set X}
    (hs : IsLocallyClosed s) (hf : LocallyIntegrableOn f s μ) (hg : ContinuousOn g s) :
    LocallyIntegrableOn (fun x => f x • g x) s μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : LocallyCompactSpace X
    inst✝³ : T2Space X
    𝕜 : Type u_6
    inst✝² : NormedField 𝕜
    inst✝¹ : SecondCountableTopologyEither X E
    inst✝ : NormedSpace 𝕜 E
    f : X → 𝕜
    g : X → E
    s : Set X
    hs : IsLocallyClosed s
    hf : MeasureTheory.LocallyIntegrableOn f s μ
    hg : ContinuousOn g s
    ⊢ MeasureTheory.LocallyIntegrableOn (fun x => HSMul.hSMul (f x) (g x)) s μ
  -/
  rw [MeasureTheory.locallyIntegrableOn_iff hs] at hf ⊢
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : LocallyCompactSpace X
    inst✝³ : T2Space X
    𝕜 : Type u_6
    inst✝² : NormedField 𝕜
    inst✝¹ : SecondCountableTopologyEither X E
    inst✝ : NormedSpace 𝕜 E
    f : X → 𝕜
    g : X → E
    s : Set X
    hs : IsLocallyClosed s
    hf : ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integra …
    hg : ContinuousOn g s
    ⊢ ∀ (k : Set X), HasSubset.Subset k s → IsCompact k → MeasureTheory.Integrable …
  -/
  exact fun k hk_sub hk_c => (hf k hk_sub hk_c).smul_continuousOn (hg.mono hk_sub) hk_c
  /-
    🎉 no goals
  -/


