instance : TopologicalSpace (List α) :=
  TopologicalSpace.mkOfNhds (traverse nhds)


theorem nhds_list (as : List α) : 𝓝 as = traverse 𝓝 as := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    as : List α
    ⊢ Eq (nhds as) (Traversable.traverse nhds as)
  -/
  refine nhds_mkOfNhds _ _ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : TopologicalSpace α
      as : List α
      ⊢ LE.le Pure.pure (Traversable.traverse nhds)
    -/
  · intro l
    induction l with
    | nil => exact le_rfl
    | cons a l ih =>
      suffices List.cons <$> pure a <*> pure l ≤ List.cons <$> 𝓝 a <*> traverse 𝓝 l by
        simpa only [functor_norm] using this
      exact Filter.seq_mono (Filter.map_mono <| pure_le_nhds a) ih
    /-
      case refine_2
      α : Type u_1
      inst✝ : TopologicalSpace α
      as : List α
      ⊢ ∀ (a : List α) (s : Set (List α)), Membership.mem (Traversable.traverse nhds …
    -/
  · intro l s hs
    /-
      case refine_2
      α : Type u_1
      inst✝ : TopologicalSpace α
      as l : List α
      s : Set (List α)
      hs : Membership.mem (Traversable.traverse nhds l) s
      ⊢ Filter.Eventually (fun y => Membership.mem (Traversable.traverse nhds y) s)  …
    -/
    rcases (mem_traverse_iff _ _).1 hs with ⟨u, hu, hus⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      as l : List α
      s : Set (List α)
      hs : Membership.mem (Traversable.traverse nhds l) s
      u : List (Set α)
      hu : List.Forall₂ (fun b s => Membership.mem (nhds b) s) l u
      hus : HasSubset.Subset (sequence u) s
      ⊢ Filter.Eventually (fun y => Membership.mem (Traversable.traverse nhds y) s)  …
    -/
    clear as hs
    have : ∃ v : List (Set α), l.Forall₂ (fun a s => IsOpen s ∧ a ∈ s) v ∧ sequence v ⊆ s := by
      induction hu generalizing s with
      | nil =>
        exists []
        simp only [List.forall₂_nil_left_iff, exists_eq_left]
        exact ⟨trivial, hus⟩
      -- porting note -- renamed reordered variables based on previous types
      | cons ht _ ih =>
        rcases mem_nhds_iff.1 ht with ⟨u, hut, hu⟩
        rcases ih _ Subset.rfl with ⟨v, hv, hvss⟩
        exact
          ⟨u::v, List.Forall₂.cons hu hv,
            Subset.trans (Set.seq_mono (Set.image_subset _ hut) hvss) hus⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      s : Set (List α)
      u : List (Set α)
      hu : List.Forall₂ (fun b s => Membership.mem (nhds b) s) l u
      hus : HasSubset.Subset (sequence u) s
      this : Exists fun v => And (List.Forall₂ (fun a s => And (IsOpen s) (Membershi …
      ⊢ Filter.Eventually (fun y => Membership.mem (Traversable.traverse nhds y) s)  …
    -/
    rcases this with ⟨v, hv, hvs⟩
    have : sequence v ∈ traverse 𝓝 l :=
      mem_traverse _ _ <| hv.imp fun a s ⟨hs, ha⟩ => IsOpen.mem_nhds hs ha
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      s : Set (List α)
      u : List (Set α)
      hu : List.Forall₂ (fun b s => Membership.mem (nhds b) s) l u
      hus : HasSubset.Subset (sequence u) s
      v : List (Set α)
      hv : List.Forall₂ (fun a s => And (IsOpen s) (Membership.mem s a)) l v
      hvs : HasSubset.Subset (sequence v) s
      this : Membership.mem (Traversable.traverse nhds l) (sequence v)
      ⊢ Filter.Eventually (fun y => Membership.mem (Traversable.traverse nhds y) s)  …
    -/
    refine mem_of_superset this fun u hu ↦ ?_
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      s : Set (List α)
      u✝ : List (Set α)
      hu✝ : List.Forall₂ (fun b s => Membership.mem (nhds b) s) l u✝
      hus : HasSubset.Subset (sequence u✝) s
      v : List (Set α)
      hv : List.Forall₂ (fun a s => And (IsOpen s) (Membership.mem s a)) l v
      hvs : HasSubset.Subset (sequence v) s
      this : Membership.mem (Traversable.traverse nhds l) (sequence v)
      u : List α
      hu : Membership.mem (sequence v) u
      ⊢ Membership.mem (setOf fun x => (fun y => Membership.mem (Traversable.travers …
    -/
    have hu := (List.mem_traverse _ _).1 hu
    have : List.Forall₂ (fun a s => IsOpen s ∧ a ∈ s) u v := by
      refine List.Forall₂.flip ?_
      replace hv := hv.flip
      #adaptation_note /-- nightly-2024-03-16: simp was
      simp only [List.forall₂_and_left, flip] at hv ⊢ -/
      simp only [List.forall₂_and_left, Function.flip_def] at hv ⊢
      exact ⟨hv.1, hu.flip⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      s : Set (List α)
      u✝ : List (Set α)
      hu✝¹ : List.Forall₂ (fun b s => Membership.mem (nhds b) s) l u✝
      hus : HasSubset.Subset (sequence u✝) s
      v : List (Set α)
      hv : List.Forall₂ (fun a s => And (IsOpen s) (Membership.mem s a)) l v
      hvs : HasSubset.Subset (sequence v) s
      this✝ : Membership.mem (Traversable.traverse nhds l) (sequence v)
      u : List α
      hu✝ : Membership.mem (sequence v) u
      hu : List.Forall₂ (fun b a => Membership.mem (id a) b) u v
      this : List.Forall₂ (fun a s => And (IsOpen s) (Membership.mem s a)) u v
      ⊢ Membership.mem (setOf fun x => (fun y => Membership.mem (Traversable.travers …
    -/
    refine mem_of_superset ?_ hvs
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      s : Set (List α)
      u✝ : List (Set α)
      hu✝¹ : List.Forall₂ (fun b s => Membership.mem (nhds b) s) l u✝
      hus : HasSubset.Subset (sequence u✝) s
      v : List (Set α)
      hv : List.Forall₂ (fun a s => And (IsOpen s) (Membership.mem s a)) l v
      hvs : HasSubset.Subset (sequence v) s
      this✝ : Membership.mem (Traversable.traverse nhds l) (sequence v)
      u : List α
      hu✝ : Membership.mem (sequence v) u
      hu : List.Forall₂ (fun b a => Membership.mem (id a) b) u v
      this : List.Forall₂ (fun a s => And (IsOpen s) (Membership.mem s a)) u v
      ⊢ Membership.mem (Traversable.traverse nhds u) (sequence v)
    -/
    exact mem_traverse _ _ (this.imp fun a s ⟨hs, ha⟩ => IsOpen.mem_nhds hs ha)
    /-
      🎉 no goals
    -/


@[simp]
theorem nhds_nil : 𝓝 ([] : List α) = pure [] := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Eq (nhds List.nil) (Pure.pure List.nil)
  -/
  rw [nhds_list, List.traverse_nil _]
  /-
    🎉 no goals
  -/


theorem nhds_cons (a : α) (l : List α) : 𝓝 (a::l) = List.cons <$> 𝓝 a <*> 𝓝 l := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    l : List α
    ⊢ Eq (nhds (List.cons a l)) (Seq.seq (Functor.map List.cons (nhds a)) fun x => …
  -/
  rw [nhds_list, List.traverse_cons _, ← nhds_list]
  /-
    🎉 no goals
  -/


theorem List.tendsto_cons {a : α} {l : List α} :
    Tendsto (fun p : α × List α => List.cons p.1 p.2) (𝓝 a ×ˢ 𝓝 l) (𝓝 (a::l)) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    a : α
    l : List α
    ⊢ Filter.Tendsto (fun p => List.cons p.1 p.2) (SProd.sprod (nhds a) (nhds l))  …
  -/
  rw [nhds_cons, Tendsto, Filter.map_prod]; exact le_rfl
                                            /-
                                              🎉 no goals
                                            -/


theorem Filter.Tendsto.cons {α : Type*} {f : α → β} {g : α → List β} {a : Filter α} {b : β}
    {l : List β} (hf : Tendsto f a (𝓝 b)) (hg : Tendsto g a (𝓝 l)) :
    Tendsto (fun a => List.cons (f a) (g a)) a (𝓝 (b::l)) :=
  List.tendsto_cons.comp (Tendsto.prod_mk hf hg)


theorem tendsto_cons_iff {β : Type*} {f : List α → β} {b : Filter β} {a : α} {l : List α} :
    Tendsto f (𝓝 (a::l)) b ↔ Tendsto (fun p : α × List α => f (p.1::p.2)) (𝓝 a ×ˢ 𝓝 l) b := by
  have : 𝓝 (a::l) = (𝓝 a ×ˢ 𝓝 l).map fun p : α × List α => p.1::p.2 := by
    simp only [nhds_cons, Filter.prod_eq, (Filter.map_def _ _).symm,
      (Filter.seq_eq_filter_seq _ _).symm]
    simp [-Filter.map_def, Function.comp_def, functor_norm]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    β : Type u_3
    f : List α → β
    b : Filter β
    a : α
    l : List α
    this : Eq (nhds (List.cons a l)) (Filter.map (fun p => List.cons p.1 p.2) (SPr …
    ⊢ Iff (Filter.Tendsto f (nhds (List.cons a l)) b) (Filter.Tendsto (fun p => f  …
  -/
  rw [this, Filter.tendsto_map'_iff]; rfl
                                      /-
                                        🎉 no goals
                                      -/


theorem continuous_cons : Continuous fun x : α × List α => (x.1::x.2 : List α) :=
  continuous_iff_continuousAt.mpr fun ⟨_x, _y⟩ => continuousAt_fst.cons continuousAt_snd


theorem tendsto_nhds {β : Type*} {f : List α → β} {r : List α → Filter β}
    (h_nil : Tendsto f (pure []) (r []))
    (h_cons :
      ∀ l a,
        Tendsto f (𝓝 l) (r l) →
          Tendsto (fun p : α × List α => f (p.1::p.2)) (𝓝 a ×ˢ 𝓝 l) (r (a::l))) :
    ∀ l, Tendsto f (𝓝 l) (r l)
             /-
               α : Type u_1
               inst✝ : TopologicalSpace α
               β : Type u_3
               f : List α → β
               r : List α → Filter β
               h_nil : Filter.Tendsto f (Pure.pure List.nil) (r List.nil)
               h_cons : ∀ (l : List α) (a : α), Filter.Tendsto f (nhds l) (r l) → Filter.Tend …
               ⊢ Filter.Tendsto f (nhds List.nil) (r List.nil)
             -/
  | [] => by rwa [nhds_nil]
             /-
               🎉 no goals
             -/
  | a::l => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      β : Type u_3
      f : List α → β
      r : List α → Filter β
      h_nil : Filter.Tendsto f (Pure.pure List.nil) (r List.nil)
      h_cons : ∀ (l : List α) (a : α), Filter.Tendsto f (nhds l) (r l) → Filter.Tend …
      a : α
      l : List α
      ⊢ Filter.Tendsto f (nhds (List.cons a l)) (r (List.cons a l))
    -/
    rw [tendsto_cons_iff]; exact h_cons l a (@tendsto_nhds _ _ _ h_nil h_cons l)
                           /-
                             🎉 no goals
                           -/


instance [DiscreteTopology α] : DiscreteTopology (List α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : DiscreteTopology α
    ⊢ DiscreteTopology (List α)
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  rw [discreteTopology_iff_nhds]; intro l; induction l <;> simp [*, nhds_cons]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem continuousAt_length : ∀ l : List α, ContinuousAt List.length l := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ ∀ (l : List α), ContinuousAt List.length l
  -/
  simp only [ContinuousAt, nhds_discrete]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ ∀ (l : List α), Filter.Tendsto List.length (nhds l) (Pure.pure l.length)
  -/
  refine tendsto_nhds ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : TopologicalSpace α
      ⊢ Filter.Tendsto List.length (Pure.pure List.nil) (Pure.pure List.nil.length)
    -/
  · exact tendsto_pure_pure _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : TopologicalSpace α
      ⊢ ∀ (l : List α) (a : α), Filter.Tendsto List.length (nhds l) (Pure.pure l.len …
    -/
  · intro l a ih
    /-
      case refine_2
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      a : α
      ih : Filter.Tendsto List.length (nhds l) (Pure.pure l.length)
      ⊢ Filter.Tendsto (fun p => (List.cons p.1 p.2).length) (SProd.sprod (nhds a) ( …
    -/
    dsimp only [List.length]
    /-
      case refine_2
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      a : α
      ih : Filter.Tendsto List.length (nhds l) (Pure.pure l.length)
      ⊢ Filter.Tendsto (fun p => HAdd.hAdd p.2.length 1) (SProd.sprod (nhds a) (nhds …
    -/
    refine Tendsto.comp (tendsto_pure_pure (fun x => x + 1) _) ?_
    /-
      case refine_2
      α : Type u_1
      inst✝ : TopologicalSpace α
      l : List α
      a : α
      ih : Filter.Tendsto List.length (nhds l) (Pure.pure l.length)
      ⊢ Filter.Tendsto (fun p => p.2.length.add 0) (SProd.sprod (nhds a) (nhds l)) ( …
    -/
    exact Tendsto.comp ih tendsto_snd
    /-
      🎉 no goals
    -/


/-- Continuity of `insertIdx` in terms of `Tendsto`. -/
theorem tendsto_insertIdx' {a : α} :
    ∀ {n : ℕ} {l : List α},
      Tendsto (fun p : α × List α => insertIdx n p.1 p.2) (𝓝 a ×ˢ 𝓝 l) (𝓝 (insertIdx n a l))
  | 0, _ => tendsto_cons
                    /-
                      α : Type u_1
                      inst✝ : TopologicalSpace α
                      a : α
                      n : Nat
                      ⊢ Filter.Tendsto (fun p => List.insertIdx (HAdd.hAdd n 1) p.1 p.2) (SProd.spro …
                    -/
  | n + 1, [] => by simp
                    /-
                      🎉 no goals
                    -/
  | n + 1, a'::l => by
    have : 𝓝 a ×ˢ 𝓝 (a'::l) =
        (𝓝 a ×ˢ (𝓝 a' ×ˢ 𝓝 l)).map fun p : α × α × List α => (p.1, p.2.1::p.2.2) := by
      simp only [nhds_cons, Filter.prod_eq, ← Filter.map_def, ← Filter.seq_eq_filter_seq]
      simp [-Filter.map_def, Function.comp_def, functor_norm]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      a : α
      n : Nat
      a' : α
      l : List α
      this : Eq (SProd.sprod (nhds a) (nhds (List.cons a' l))) (Filter.map (fun p => …
      ⊢ Filter.Tendsto (fun p => List.insertIdx (HAdd.hAdd n 1) p.1 p.2) (SProd.spro …
    -/
    rw [this, tendsto_map'_iff]
    exact
      (tendsto_fst.comp tendsto_snd).cons
        ((@tendsto_insertIdx' _ n l).comp <| tendsto_fst.prod_mk <| tendsto_snd.comp tendsto_snd)


@[deprecated (since := "2024-10-21")] alias tendsto_insertNth' := tendsto_insertIdx'


theorem tendsto_insertIdx {β} {n : ℕ} {a : α} {l : List α} {f : β → α} {g : β → List α}
    {b : Filter β} (hf : Tendsto f b (𝓝 a)) (hg : Tendsto g b (𝓝 l)) :
    Tendsto (fun b : β => insertIdx n (f b) (g b)) b (𝓝 (insertIdx n a l)) :=
  tendsto_insertIdx'.comp (Tendsto.prod_mk hf hg)


@[deprecated (since := "2024-10-21")] alias tendsto_insertNth := tendsto_insertIdx'


theorem continuous_insertIdx {n : ℕ} : Continuous fun p : α × List α => insertIdx n p.1 p.2 :=
  continuous_iff_continuousAt.mpr fun ⟨a, l⟩ => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      x✝ : Prod α (List α)
      a : α
      l : List α
      ⊢ ContinuousAt (fun p => List.insertIdx n p.1 p.2) { fst := a, snd := l }
    -/
    rw [ContinuousAt, nhds_prod_eq]; exact tendsto_insertIdx'
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-10-21")] alias continuous_insertNth := continuous_insertIdx


theorem tendsto_eraseIdx :
    ∀ {n : ℕ} {l : List α}, Tendsto (eraseIdx · n) (𝓝 l) (𝓝 (eraseIdx l n))
                /-
                  α : Type u_1
                  inst✝ : TopologicalSpace α
                  x✝ : Nat
                  ⊢ Filter.Tendsto (fun x => x.eraseIdx x✝) (nhds List.nil) (nhds (List.nil.eras …
                -/
  | _, [] => by rw [nhds_nil]; exact tendsto_pure_nhds _ _
                               /-
                                 🎉 no goals
                               -/
                  /-
                    α : Type u_1
                    inst✝ : TopologicalSpace α
                    a : α
                    l : List α
                    ⊢ Filter.Tendsto (fun x => x.eraseIdx 0) (nhds (List.cons a l)) (nhds ((List.c …
                  -/
  | 0, a::l => by rw [tendsto_cons_iff]; exact tendsto_snd
                                         /-
                                           🎉 no goals
                                         -/
  | n + 1, a::l => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      a : α
      l : List α
      ⊢ Filter.Tendsto (fun x => x.eraseIdx (HAdd.hAdd n 1)) (nhds (List.cons a l))  …
    -/
    rw [tendsto_cons_iff]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      a : α
      l : List α
      ⊢ Filter.Tendsto (fun p => (List.cons p.1 p.2).eraseIdx (HAdd.hAdd n 1)) (SPro …
    -/
    dsimp [eraseIdx]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      a : α
      l : List α
      ⊢ Filter.Tendsto (fun p => List.cons p.1 (p.2.eraseIdx n)) (SProd.sprod (nhds  …
    -/
    exact tendsto_fst.cons ((@tendsto_eraseIdx n l).comp tendsto_snd)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-05-04")] alias tendsto_removeNth := tendsto_eraseIdx


theorem continuous_eraseIdx {n : ℕ} : Continuous fun l : List α => eraseIdx l n :=
  continuous_iff_continuousAt.mpr fun _a => tendsto_eraseIdx


@[deprecated (since := "2024-05-04")] alias continuous_removeNth := continuous_eraseIdx


@[to_additive]
theorem tendsto_prod [Monoid α] [ContinuousMul α] {l : List α} :
    Tendsto List.prod (𝓝 l) (𝓝 l.prod) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Monoid α
    inst✝ : ContinuousMul α
    l : List α
    ⊢ Filter.Tendsto List.prod (nhds l) (nhds l.prod)
  -/
  induction' l with x l ih
    /-
      case nil
      α : Type u_1
      inst✝² : TopologicalSpace α
      inst✝¹ : Monoid α
      inst✝ : ContinuousMul α
      ⊢ Filter.Tendsto List.prod (nhds List.nil) (nhds List.nil.prod)
    -/
  · simp +contextual [nhds_nil, mem_of_mem_nhds, tendsto_pure_left]
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Monoid α
    inst✝ : ContinuousMul α
    x : α
    l : List α
    ih : Filter.Tendsto List.prod (nhds l) (nhds l.prod)
    ⊢ Filter.Tendsto List.prod (nhds (List.cons x l)) (nhds (List.cons x l).prod)
  -/
  simp_rw [tendsto_cons_iff, prod_cons]
  /-
    case cons
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Monoid α
    inst✝ : ContinuousMul α
    x : α
    l : List α
    ih : Filter.Tendsto List.prod (nhds l) (nhds l.prod)
    ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2.prod) (SProd.sprod (nhds x) (nhds …
  -/
  have := continuous_iff_continuousAt.mp continuous_mul (x, l.prod)
  /-
    case cons
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Monoid α
    inst✝ : ContinuousMul α
    x : α
    l : List α
    ih : Filter.Tendsto List.prod (nhds l) (nhds l.prod)
    this : ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := x, snd := l.prod }
    ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2.prod) (SProd.sprod (nhds x) (nhds …
  -/
  rw [ContinuousAt, nhds_prod_eq] at this
  /-
    case cons
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : Monoid α
    inst✝ : ContinuousMul α
    x : α
    l : List α
    ih : Filter.Tendsto List.prod (nhds l) (nhds l.prod)
    this : Filter.Tendsto (fun p => HMul.hMul p.1 p.2) (SProd.sprod (nhds x) (nhds …
    ⊢ Filter.Tendsto (fun p => HMul.hMul p.1 p.2.prod) (SProd.sprod (nhds x) (nhds …
  -/
  exact this.comp (tendsto_id.prod_map ih)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem continuous_prod [Monoid α] [ContinuousMul α] : Continuous (prod : List α → α) :=
  continuous_iff_continuousAt.mpr fun _l => tendsto_prod


                                                            /-
                                                              α : Type u_1
                                                              β : Type u_2
                                                              inst✝¹ : TopologicalSpace α
                                                              inst✝ : TopologicalSpace β
                                                              n : Nat
                                                              ⊢ TopologicalSpace (List.Vector α n)
                                                            -/
instance (n : ℕ) : TopologicalSpace (List.Vector α n) := by unfold List.Vector; infer_instance
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem tendsto_cons {n : ℕ} {a : α} {l : List.Vector α n} :
    Tendsto (fun p : α × List.Vector α n => p.1 ::ᵥ p.2) (𝓝 a ×ˢ 𝓝 l) (𝓝 (a ::ᵥ l)) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    n : Nat
    a : α
    l : List.Vector α n
    ⊢ Filter.Tendsto (fun p => List.Vector.cons p.1 p.2) (SProd.sprod (nhds a) (nh …
  -/
  rw [tendsto_subtype_rng, Vector.cons_val]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    n : Nat
    a : α
    l : List.Vector α n
    ⊢ Filter.Tendsto (fun x => ↑(List.Vector.cons x.1 x.2)) (SProd.sprod (nhds a)  …
  -/
  exact tendsto_fst.cons (Tendsto.comp continuousAt_subtype_val tendsto_snd)
  /-
    🎉 no goals
  -/


theorem tendsto_insertIdx {n : ℕ} {i : Fin (n + 1)} {a : α} :
    ∀ {l : List.Vector α n},
      Tendsto (fun p : α × List.Vector α n => Vector.insertIdx p.1 i p.2) (𝓝 a ×ˢ 𝓝 l)
        (𝓝 (Vector.insertIdx a i l))
  | ⟨l, hl⟩ => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a : α
      l : List α
      hl : Eq l.length n
      ⊢ Filter.Tendsto (fun p => List.Vector.insertIdx p.1 i p.2) (SProd.sprod (nhds …
    -/
    rw [Vector.insertIdx, tendsto_subtype_rng]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a : α
      l : List α
      hl : Eq l.length n
      ⊢ Filter.Tendsto (fun x => ↑(List.Vector.insertIdx x.1 i x.2)) (SProd.sprod (n …
    -/
    simp only [Vector.insertIdx_val]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a : α
      l : List α
      hl : Eq l.length n
      ⊢ Filter.Tendsto (fun x => List.insertIdx (↑i) x.1 ↑x.2) (SProd.sprod (nhds a) …
    -/
    exact List.tendsto_insertIdx tendsto_fst (Tendsto.comp continuousAt_subtype_val tendsto_snd : _)
    /-
      🎉 no goals
    -/


/-- Continuity of `Vector.insertIdx`. -/
theorem continuous_insertIdx' {n : ℕ} {i : Fin (n + 1)} :
    Continuous fun p : α × List.Vector α n => Vector.insertIdx p.1 i p.2 :=
  continuous_iff_continuousAt.mpr fun ⟨a, l⟩ => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      x✝ : Prod α (List.Vector α n)
      a : α
      l : List.Vector α n
      ⊢ ContinuousAt (fun p => List.Vector.insertIdx p.1 i p.2) { fst := a, snd := l }
    -/
    rw [ContinuousAt, nhds_prod_eq]; exact tendsto_insertIdx
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-10-21")] alias continuous_insertNth' := continuous_insertIdx'


theorem continuous_insertIdx {n : ℕ} {i : Fin (n + 1)} {f : β → α} {g : β → List.Vector α n}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun b => Vector.insertIdx (f b) i (g b) :=
  continuous_insertIdx'.comp (hf.prod_mk hg : _)


theorem continuousAt_eraseIdx {n : ℕ} {i : Fin (n + 1)} :
    ∀ {l : List.Vector α (n + 1)}, ContinuousAt (List.Vector.eraseIdx i) l
  | ⟨l, hl⟩ => by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      l : List α
      hl : Eq l.length (HAdd.hAdd n 1)
      ⊢ ContinuousAt (List.Vector.eraseIdx i) ⟨l, hl⟩
    -/
    rw [ContinuousAt, List.Vector.eraseIdx, tendsto_subtype_rng]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      l : List α
      hl : Eq l.length (HAdd.hAdd n 1)
      ⊢ Filter.Tendsto (fun x => ↑(List.Vector.eraseIdx i x)) (nhds ⟨l, hl⟩) (nhds ↑ …
    -/
    simp only [Vector.eraseIdx_val]
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      l : List α
      hl : Eq l.length (HAdd.hAdd n 1)
      ⊢ Filter.Tendsto (fun x => (↑x).eraseIdx ↑i) (nhds ⟨l, hl⟩) (nhds (l.eraseIdx  …
    -/
    exact Tendsto.comp List.tendsto_eraseIdx continuousAt_subtype_val
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-05-04")] alias continuousAt_removeNth := continuousAt_eraseIdx


theorem continuous_eraseIdx {n : ℕ} {i : Fin (n + 1)} :
    Continuous (List.Vector.eraseIdx i : List.Vector α (n + 1) → List.Vector α n) :=
  continuous_iff_continuousAt.mpr fun ⟨_a, _l⟩ => continuousAt_eraseIdx


