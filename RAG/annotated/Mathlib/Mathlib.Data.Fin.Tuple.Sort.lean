/-- `graph f` produces the finset of pairs `(f i, i)`
equipped with the lexicographic order.
-/
def graph (f : Fin n → α) : Finset (α ×ₗ Fin n) :=
  Finset.univ.image fun i => (f i, i)


/-- Given `p : α ×ₗ (Fin n) := (f i, i)` with `p ∈ graph f`,
`graph.proj p` is defined to be `f i`.
-/
def graph.proj {f : Fin n → α} : graph f → α := fun p => p.1.1


@[simp]
theorem graph.card (f : Fin n → α) : (graph f).card = n := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Eq (Tuple.graph f).card n
  -/
  rw [graph, Finset.card_image_of_injective]
    /-
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      ⊢ Eq Finset.univ.card n
    -/
  · exact Finset.card_fin _
    /-
      🎉 no goals
    -/
    /-
      case H
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      ⊢ Function.Injective fun i => { fst := f i, snd := i }
    -/
  · intro _ _
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp`
    /-
      case H
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      a₁✝ a₂✝ : Fin n
      ⊢ Eq ((fun i => { fst := f i, snd := i }) a₁✝) ((fun i => { fst := f i, snd := …
    -/
    dsimp only
    /-
      case H
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      a₁✝ a₂✝ : Fin n
      ⊢ Eq { fst := f a₁✝, snd := a₁✝ } { fst := f a₂✝, snd := a₂✝ } → Eq a₁✝ a₂✝
    -/
    rw [Prod.ext_iff]
    /-
      case H
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      a₁✝ a₂✝ : Fin n
      ⊢ And (Eq { fst := f a₁✝, snd := a₁✝ }.1 { fst := f a₂✝, snd := a₂✝ }.1) (Eq { …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `graphEquiv₁ f` is the natural equivalence between `Fin n` and `graph f`,
mapping `i` to `(f i, i)`. -/
def graphEquiv₁ (f : Fin n → α) : Fin n ≃ graph f where
                           /-
                             n : Nat
                             α : Type u_1
                             inst✝ : LinearOrder α
                             f : Fin n → α
                             i : Fin n
                             ⊢ Membership.mem (Tuple.graph f) { fst := f i, snd := i }
                           -/
  toFun i := ⟨(f i, i), by simp [graph]⟩
                           /-
                             🎉 no goals
                           -/
  invFun p := p.1.2
                   /-
                     n : Nat
                     α : Type u_1
                     inst✝ : LinearOrder α
                     f : Fin n → α
                     i : Fin n
                     ⊢ Eq ((fun p => (↑p).2) ((fun i => ⟨{ fst := f i, snd := i }, ⋯⟩) i)) i
                   -/
  left_inv i := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv := fun ⟨⟨x, i⟩, h⟩ => by
    -- Porting note: was `simpa [graph] using h`
    /-
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      x✝ : Subtype fun x => Membership.mem (Tuple.graph f) x
      x : α
      i : Fin n
      h : Membership.mem (Tuple.graph f) { fst := x, snd := i }
      ⊢ Eq ((fun i => ⟨{ fst := f i, snd := i }, ⋯⟩) ((fun p => (↑p).2) ⟨{ fst := x, …
    -/
    simp only [graph, Finset.mem_image, Finset.mem_univ, true_and] at h
    /-
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      x✝ : Subtype fun x => Membership.mem (Tuple.graph f) x
      x : α
      i : Fin n
      h✝ : Membership.mem (Tuple.graph f) { fst := x, snd := i }
      h : Exists fun a => Eq { fst := f a, snd := a } { fst := x, snd := i }
      ⊢ Eq ((fun i => ⟨{ fst := f i, snd := i }, ⋯⟩) ((fun p => (↑p).2) ⟨{ fst := x, …
    -/
    obtain ⟨i', hi'⟩ := h
    /-
      case intro
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      x✝ : Subtype fun x => Membership.mem (Tuple.graph f) x
      x : α
      i : Fin n
      h : Membership.mem (Tuple.graph f) { fst := x, snd := i }
      i' : Fin n
      hi' : Eq { fst := f i', snd := i' } { fst := x, snd := i }
      ⊢ Eq ((fun i => ⟨{ fst := f i, snd := i }, ⋯⟩) ((fun p => (↑p).2) ⟨{ fst := x, …
    -/
    obtain ⟨-, rfl⟩ := Prod.mk.inj_iff.mp hi'
    /-
      case intro.intro
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      x✝ : Subtype fun x => Membership.mem (Tuple.graph f) x
      x : α
      i' : Fin n
      h : Membership.mem (Tuple.graph f) { fst := x, snd := i' }
      hi' : Eq { fst := f i', snd := i' } { fst := x, snd := i' }
      ⊢ Eq ((fun i => ⟨{ fst := f i, snd := i }, ⋯⟩) ((fun p => (↑p).2) ⟨{ fst := x, …
    -/
    simpa
    /-
      🎉 no goals
    -/


@[simp]
theorem proj_equiv₁' (f : Fin n → α) : graph.proj ∘ graphEquiv₁ f = f :=
  rfl


/-- `graphEquiv₂ f` is an equivalence between `Fin n` and `graph f` that respects the order.
-/
def graphEquiv₂ (f : Fin n → α) : Fin n ≃o graph f :=
                             /-
                               n : Nat
                               α : Type u_1
                               inst✝ : LinearOrder α
                               f : Fin n → α
                               ⊢ Eq (Tuple.graph f).card n
                             -/
  Finset.orderIsoOfFin _ (by simp)
                             /-
                               🎉 no goals
                             -/


/-- `sort f` is the permutation that orders `Fin n` according to the order of the outputs of `f`. -/
def sort (f : Fin n → α) : Equiv.Perm (Fin n) :=
  (graphEquiv₂ f).toEquiv.trans (graphEquiv₁ f).symm


theorem graphEquiv₂_apply (f : Fin n → α) (i : Fin n) :
    graphEquiv₂ f i = graphEquiv₁ f (sort f i) :=
  ((graphEquiv₁ f).apply_symm_apply _).symm


theorem self_comp_sort (f : Fin n → α) : f ∘ sort f = graph.proj ∘ graphEquiv₂ f :=
                                                                                            /-
                                                                                              n : Nat
                                                                                              α : Type u_1
                                                                                              inst✝ : LinearOrder α
                                                                                              f : Fin n → α
                                                                                              ⊢ Eq (Function.comp Tuple.graph.proj (Function.comp (Function.comp ⇑(Tuple.gra …
                                                                                            -/
  show graph.proj ∘ (graphEquiv₁ f ∘ (graphEquiv₁ f).symm) ∘ (graphEquiv₂ f).toEquiv = _ by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


theorem monotone_proj (f : Fin n → α) : Monotone (graph.proj : graph f → α) := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Monotone Tuple.graph.proj
  -/
  rintro ⟨⟨x, i⟩, hx⟩ ⟨⟨y, j⟩, hy⟩ (_ | h)
    /-
      case mk.mk.mk.mk.left
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      x : α
      i : Fin n
      hx : Membership.mem (Tuple.graph f) { fst := x, snd := i }
      y : α
      j : Fin n
      hy : Membership.mem (Tuple.graph f) { fst := y, snd := j }
      h✝ : LT.lt x y
      ⊢ LE.le (Tuple.graph.proj ⟨{ fst := x, snd := i }, hx⟩) (Tuple.graph.proj ⟨{ f …
    -/
  · exact le_of_lt ‹_›
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.mk.mk.right
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      x : α
      i : Fin n
      hx : Membership.mem (Tuple.graph f) { fst := x, snd := i }
      j : Fin n
      hy : Membership.mem (Tuple.graph f) { fst := x, snd := j }
      h✝ : LE.le i j
      ⊢ LE.le (Tuple.graph.proj ⟨{ fst := x, snd := i }, hx⟩) (Tuple.graph.proj ⟨{ f …
    -/
  · simp [graph.proj]
    /-
      🎉 no goals
    -/


theorem monotone_sort (f : Fin n → α) : Monotone (f ∘ sort f) := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Monotone (Function.comp f ⇑(Tuple.sort f))
  -/
  rw [self_comp_sort]
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Monotone (Function.comp Tuple.graph.proj ⇑(Tuple.graphEquiv₂ f))
  -/
  exact (monotone_proj f).comp (graphEquiv₂ f).monotone
  /-
    🎉 no goals
  -/


/-- If `f₀ ≤ f₁ ≤ f₂ ≤ ⋯` is a sorted `m`-tuple of elements of `α`, then for any `j : Fin m` and
`a : α` we have `j < #{i | fᵢ ≤ a}` iff `fⱼ ≤ a`. -/
theorem lt_card_le_iff_apply_le_of_monotone [PartialOrder α] [DecidableRel (α := α) LE.le]
    {m : ℕ} (f : Fin m → α) (a : α) (h_sorted : Monotone f) (j : Fin m) :
    j < Fintype.card {i // f i ≤ a} ↔ f j ≤ a := by
  suffices h1 : ∀ k : Fin m, (k < Fintype.card {i // f i ≤ a}) → f k ≤ a by
    refine ⟨h1 j, fun h ↦ ?_⟩
    by_contra! hc
    let p : Fin m → Prop := fun x ↦ f x ≤ a
    let q : Fin m → Prop := fun x ↦ x < Fintype.card {i // f i ≤ a}
    let q' : {i // f i ≤ a} → Prop := fun x ↦ q x
    have hw : 0 < Fintype.card {j : {x : Fin m // f x ≤ a} // ¬ q' j} :=
      Fintype.card_pos_iff.2 ⟨⟨⟨j, h⟩, not_lt.2 hc⟩⟩
    apply hw.ne'
    have he := Fintype.card_congr <| Equiv.sumCompl <| q'
    have h4 := (Fintype.card_congr (@Equiv.subtypeSubtypeEquivSubtype _ p q (h1 _)))
    have h_le : Fintype.card { i // f i ≤ a } ≤ m := by
      conv_rhs => rw [← Fintype.card_fin m]
      exact Fintype.card_subtype_le _
    rwa [Fintype.card_sum, h4, Fintype.card_fin_lt_of_le h_le, add_right_eq_self] at he
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j : Fin m
    ⊢ ∀ (k : Fin m), LT.lt (↑k) (Fintype.card (Subtype fun i => LE.le (f i) a)) →  …
  -/
  intro _ h
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : LT.lt (↑k✝) (Fintype.card (Subtype fun i => LE.le (f i) a))
    ⊢ LE.le (f k✝) a
  -/
  contrapose! h
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : Not (LE.le (f k✝) a)
    ⊢ LE.le (Fintype.card (Subtype fun i => LE.le (f i) a)) ↑k✝
  -/
  rw [← Fin.card_Iio, Fintype.card_subtype]
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : Not (LE.le (f k✝) a)
    ⊢ LE.le (Finset.filter (fun x => LE.le (f x) a) Finset.univ).card (Finset.Iio  …
  -/
  refine Finset.card_mono (fun i => Function.mtr ?_)
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : Not (LE.le (f k✝) a)
    i : Fin m
    ⊢ Not (Membership.mem (Finset.Iio k✝) i) → Not (Membership.mem (Finset.filter  …
  -/
  simp_rw [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iio]
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : Not (LE.le (f k✝) a)
    i : Fin m
    ⊢ Not (LT.lt i k✝) → Not (LE.le (f i) a)
  -/
  intro hij hia
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : Not (LE.le (f k✝) a)
    i : Fin m
    hij : Not (LT.lt i k✝)
    hia : LE.le (f i) a
    ⊢ False
  -/
  apply h
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : DecidableRel LE.le
    m : Nat
    f : Fin m → α
    a : α
    h_sorted : Monotone f
    j k✝ : Fin m
    h : Not (LE.le (f k✝) a)
    i : Fin m
    hij : Not (LT.lt i k✝)
    hia : LE.le (f i) a
    ⊢ LE.le (f k✝) a
  -/
  exact (h_sorted (le_of_not_lt hij)).trans hia
  /-
    🎉 no goals
  -/


theorem lt_card_ge_iff_apply_ge_of_antitone [PartialOrder α] [DecidableRel (α := α) LE.le]
    {m : ℕ} (f : Fin m → α) (a : α) (h_sorted : Antitone f) (j : Fin m) :
    j < Fintype.card {i // a ≤ f i} ↔ a ≤ f j :=
  lt_card_le_iff_apply_le_of_monotone _ (OrderDual.toDual a) h_sorted.dual_right j


/-- If two permutations of a tuple `f` are both monotone, then they are equal. -/
theorem unique_monotone [PartialOrder α] {f : Fin n → α} {σ τ : Equiv.Perm (Fin n)}
    (hfσ : Monotone (f ∘ σ)) (hfτ : Monotone (f ∘ τ)) : f ∘ σ = f ∘ τ :=
  ofFn_injective <|
    eq_of_perm_of_sorted ((σ.ofFn_comp_perm f).trans (τ.ofFn_comp_perm f).symm)
      hfσ.ofFn_sorted hfτ.ofFn_sorted


/-- A permutation `σ` equals `sort f` if and only if the map `i ↦ (f (σ i), σ i)` is
strictly monotone (w.r.t. the lexicographic ordering on the target). -/
theorem eq_sort_iff' : σ = sort f ↔ StrictMono (σ.trans <| graphEquiv₁ f) := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    ⊢ Iff (Eq σ (Tuple.sort f)) (StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f)))
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : Eq σ (Tuple.sort f)
      ⊢ StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f))
    -/
  · rw [h, sort, Equiv.trans_assoc, Equiv.symm_trans_self]
    /-
      case mp
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : Eq σ (Tuple.sort f)
      ⊢ StrictMono ⇑((Tuple.graphEquiv₂ f).trans (Equiv.refl (Subtype fun x => Membe …
    -/
    exact (graphEquiv₂ f).strictMono
    /-
      🎉 no goals
    -/
    /-
      case mpr
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f))
      ⊢ Eq σ (Tuple.sort f)
    -/
  · have := Subsingleton.elim (graphEquiv₂ f) (h.orderIsoOfSurjective _ <| Equiv.surjective _)
    /-
      case mpr
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f))
      this : Eq (Tuple.graphEquiv₂ f) (StrictMono.orderIsoOfSurjective (⇑(Equiv.tran …
      ⊢ Eq σ (Tuple.sort f)
    -/
    ext1 x
    /-
      case mpr.H
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f))
      this : Eq (Tuple.graphEquiv₂ f) (StrictMono.orderIsoOfSurjective (⇑(Equiv.tran …
      x : Fin n
      ⊢ Eq (σ x) ((Tuple.sort f) x)
    -/
    exact (graphEquiv₁ f).apply_eq_iff_eq_symm_apply.1 (DFunLike.congr_fun this x).symm
    /-
      🎉 no goals
    -/


/-- A permutation `σ` equals `sort f` if and only if `f ∘ σ` is monotone and whenever `i < j`
and `f (σ i) = f (σ j)`, then `σ i < σ j`. This means that `sort f` is the lexicographically
smallest permutation `σ` such that `f ∘ σ` is monotone. -/
theorem eq_sort_iff :
    σ = sort f ↔ Monotone (f ∘ σ) ∧ ∀ i j, i < j → f (σ i) = f (σ j) → σ i < σ j := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    ⊢ Iff (Eq σ (Tuple.sort f)) (And (Monotone (Function.comp f ⇑σ)) (∀ (i j : Fin …
  -/
  rw [eq_sort_iff']
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    ⊢ Iff (StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f))) (And (Monotone (Func …
  -/
  refine ⟨fun h => ⟨(monotone_proj f).comp h.monotone, fun i j hij hfij => ?_⟩, fun h i j hij => ?_⟩
    /-
      case refine_1
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : StrictMono ⇑(Equiv.trans σ (Tuple.graphEquiv₁ f))
      i j : Fin n
      hij : LT.lt i j
      hfij : Eq (f (σ i)) (f (σ j))
      ⊢ LT.lt (σ i) (σ j)
    -/
  · exact (((Prod.Lex.lt_iff _ _).1 <| h hij).resolve_left hfij.not_lt).2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : And (Monotone (Function.comp f ⇑σ)) (∀ (i j : Fin n), LT.lt i j → Eq (f (σ …
      i j : Fin n
      hij : LT.lt i j
      ⊢ LT.lt ((Equiv.trans σ (Tuple.graphEquiv₁ f)) i) ((Equiv.trans σ (Tuple.graph …
    -/
  · obtain he | hl := (h.1 hij.le).eq_or_lt <;> apply (Prod.Lex.lt_iff _ _).2
    /-
      case refine_2.inl
      n : Nat
      α : Type u_1
      inst✝ : LinearOrder α
      f : Fin n → α
      σ : Equiv.Perm (Fin n)
      h : And (Monotone (Function.comp f ⇑σ)) (∀ (i j : Fin n), LT.lt i j → Eq (f (σ …
      i j : Fin n
      hij : LT.lt i j
      he : Eq (Function.comp f (⇑σ) i) (Function.comp f (⇑σ) j)
      ⊢ Or (LT.lt { fst := f (σ i), snd := σ i }.1 { fst := f (σ j), snd := σ j }.1) …
    -/
    exacts [Or.inr ⟨he, h.2 i j hij he⟩, Or.inl hl]
    /-
      🎉 no goals
    -/


/-- The permutation that sorts `f` is the identity if and only if `f` is monotone. -/
theorem sort_eq_refl_iff_monotone : sort f = Equiv.refl _ ↔ Monotone f := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Iff (Eq (Tuple.sort f) (Equiv.refl (Fin n))) (Monotone f)
  -/
  rw [eq_comm, eq_sort_iff, Equiv.coe_refl, Function.comp_id]
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Iff (And (Monotone f) (∀ (i j : Fin n), LT.lt i j → Eq (f (id i)) (f (id j)) …
  -/
  simp only [id, and_iff_left_iff_imp]
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    ⊢ Monotone f → ∀ (i j : Fin n), LT.lt i j → Eq (f i) (f j) → LT.lt i j
  -/
  exact fun _ _ _ hij _ => hij
  /-
    🎉 no goals
  -/


/-- A permutation of a tuple `f` is `f` sorted if and only if it is monotone. -/
theorem comp_sort_eq_comp_iff_monotone : f ∘ σ = f ∘ sort f ↔ Monotone (f ∘ σ) :=
  ⟨fun h => h.symm ▸ monotone_sort f, fun h => unique_monotone h (monotone_sort f)⟩


/-- The sorted versions of a tuple `f` and of any permutation of `f` agree. -/
theorem comp_perm_comp_sort_eq_comp_sort : (f ∘ σ) ∘ sort (f ∘ σ) = f ∘ sort f := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    ⊢ Eq (Function.comp (Function.comp f ⇑σ) ⇑(Tuple.sort (Function.comp f ⇑σ))) ( …
  -/
  rw [Function.comp_assoc, ← Equiv.Perm.coe_mul]
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    ⊢ Eq (Function.comp f ⇑(HMul.hMul σ (Tuple.sort (Function.comp f ⇑σ)))) (Funct …
  -/
  exact unique_monotone (monotone_sort (f ∘ σ)) (monotone_sort f)
  /-
    🎉 no goals
  -/


/-- If a permutation `f ∘ σ` of the tuple `f` is not the same as `f ∘ sort f`, then `f ∘ σ`
has a pair of strictly decreasing entries. -/
theorem antitone_pair_of_not_sorted' (h : f ∘ σ ≠ f ∘ sort f) :
    ∃ i j, i < j ∧ (f ∘ σ) j < (f ∘ σ) i := by
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    h : Ne (Function.comp f ⇑σ) (Function.comp f ⇑(Tuple.sort f))
    ⊢ Exists fun i => Exists fun j => And (LT.lt i j) (LT.lt (Function.comp f (⇑σ) …
  -/
  contrapose! h
  /-
    n : Nat
    α : Type u_1
    inst✝ : LinearOrder α
    f : Fin n → α
    σ : Equiv.Perm (Fin n)
    h : ∀ (i j : Fin n), LT.lt i j → LE.le (Function.comp f (⇑σ) i) (Function.comp …
    ⊢ Eq (Function.comp f ⇑σ) (Function.comp f ⇑(Tuple.sort f))
  -/
  exact comp_sort_eq_comp_iff_monotone.mpr (monotone_iff_forall_lt.mpr h)
  /-
    🎉 no goals
  -/


/-- If the tuple `f` is not the same as `f ∘ sort f`, then `f` has a pair of strictly decreasing
entries. -/
theorem antitone_pair_of_not_sorted (h : f ≠ f ∘ sort f) : ∃ i j, i < j ∧ f j < f i :=
  antitone_pair_of_not_sorted' (id h : f ∘ Equiv.refl _ ≠ _)


