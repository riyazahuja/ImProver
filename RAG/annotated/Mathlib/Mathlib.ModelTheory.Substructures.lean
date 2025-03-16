/-- Indicates that a set in a given structure is a closed under a function symbol. -/
def ClosedUnder : Prop :=
  ∀ x : Fin n → M, (∀ i : Fin n, x i ∈ s) → funMap f x ∈ s


@[simp]
theorem closedUnder_univ : ClosedUnder f (univ : Set M) := fun _ _ => mem_univ _


theorem inter (hs : ClosedUnder f s) (ht : ClosedUnder f t) : ClosedUnder f (s ∩ t) := fun x h =>
  mem_inter (hs x fun i => mem_of_mem_inter_left (h i)) (ht x fun i => mem_of_mem_inter_right (h i))


theorem inf (hs : ClosedUnder f s) (ht : ClosedUnder f t) : ClosedUnder f (s ⊓ t) :=
  hs.inter ht


theorem sInf (hS : ∀ s, s ∈ S → ClosedUnder f s) : ClosedUnder f (sInf S) := fun x h s hs =>
  hS s hs x fun i => h i s hs


/-- A substructure of a structure `M` is a set closed under application of function symbols. -/
structure Substructure where
  carrier : Set M
  fun_mem : ∀ {n}, ∀ f : L.Functions n, ClosedUnder f carrier


instance instSetLike : SetLike (L.Substructure M) M :=
                                         /-
                                           L : FirstOrder.Language
                                           M : Type w
                                           N : Type u_1
                                           P : Type u_2
                                           inst✝² : L.Structure M
                                           inst✝¹ : L.Structure N
                                           inst✝ : L.Structure P
                                           p q : L.Substructure M
                                           h : Eq ↑p ↑q
                                           ⊢ Eq p q
                                         -/
  ⟨Substructure.carrier, fun p q h => by cases p; cases q; congr⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- See Note [custom simps projection] -/
def Simps.coe (S : L.Substructure M) : Set M :=
  S


@[simp]
theorem mem_carrier {s : L.Substructure M} {x : M} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


/-- Two substructures are equal if they have the same elements. -/
@[ext]
theorem ext {S T : L.Substructure M} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


/-- Copy a substructure replacing `carrier` with a set that is equal to it. -/
protected def copy (S : L.Substructure M) (s : Set M) (hs : s = S) : L.Substructure M where
  carrier := s
  fun_mem _ f := hs.symm ▸ S.fun_mem _ f


theorem Term.realize_mem {α : Type*} (t : L.Term α) (xs : α → M) (h : ∀ a, xs a ∈ S) :
    t.realize xs ∈ S := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    α : Type u_3
    t : L.Term α
    xs : α → M
    h : ∀ (a : α), Membership.mem S (xs a)
    ⊢ Membership.mem S (FirstOrder.Language.Term.realize xs t)
  -/
  induction' t with a n f ts ih
    /-
      case var
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      S : L.Substructure M
      α : Type u_3
      xs : α → M
      h : ∀ (a : α), Membership.mem S (xs a)
      a : α
      ⊢ Membership.mem S (FirstOrder.Language.Term.realize xs (FirstOrder.Language.T …
    -/
  · exact h a
    /-
      🎉 no goals
    -/
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      S : L.Substructure M
      α : Type u_3
      xs : α → M
      h : ∀ (a : α), Membership.mem S (xs a)
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term α
      ih : ∀ (a : Fin n), Membership.mem S (FirstOrder.Language.Term.realize xs (ts  …
      ⊢ Membership.mem S (FirstOrder.Language.Term.realize xs (FirstOrder.Language.T …
    -/
  · exact Substructure.fun_mem _ _ _ ih
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_copy {s : Set M} (hs : s = S) : (S.copy s hs : Set M) = s :=
  rfl


theorem copy_eq {s : Set M} (hs : s = S) : S.copy s hs = S :=
  SetLike.coe_injective hs


theorem constants_mem (c : L.Constants) : (c : M) ∈ S :=
  mem_carrier.2 (S.fun_mem c _ finZeroElim)


/-- The substructure `M` of the structure `M`. -/
instance instTop : Top (L.Substructure M) :=
  ⟨{  carrier := Set.univ
      fun_mem := fun {_} _ _ _ => Set.mem_univ _ }⟩


instance instInhabited : Inhabited (L.Substructure M) :=
  ⟨⊤⟩


@[simp]
theorem mem_top (x : M) : x ∈ (⊤ : L.Substructure M) :=
  Set.mem_univ x


@[simp]
theorem coe_top : ((⊤ : L.Substructure M) : Set M) = Set.univ :=
  rfl


/-- The inf of two substructures is their intersection. -/
instance instInf : Min (L.Substructure M) :=
  ⟨fun S₁ S₂ =>
    { carrier := (S₁ : Set M) ∩ (S₂ : Set M)
      fun_mem := fun {_} f => (S₁.fun_mem f).inf (S₂.fun_mem f) }⟩


@[simp]
theorem coe_inf (p p' : L.Substructure M) :
    ((p ⊓ p' : L.Substructure M) : Set M) = (p : Set M) ∩ (p' : Set M) :=
  rfl


@[simp]
theorem mem_inf {p p' : L.Substructure M} {x : M} : x ∈ p ⊓ p' ↔ x ∈ p ∧ x ∈ p' :=
  Iff.rfl


instance instInfSet : InfSet (L.Substructure M) :=
  ⟨fun s =>
    { carrier := ⋂ t ∈ s, (t : Set M)
      fun_mem := fun {n} f =>
        ClosedUnder.sInf
          (by
            /-
              L : FirstOrder.Language
              M : Type w
              N : Type u_1
              P : Type u_2
              inst✝² : L.Structure M
              inst✝¹ : L.Structure N
              inst✝ : L.Structure P
              S : L.Substructure M
              s : Set (L.Substructure M)
              n : Nat
              f : L.Functions n
              ⊢ ∀ (s_1 : Set M), Membership.mem (Set.range fun t => Set.iInter fun h => ↑t)  …
            -/
            rintro _ ⟨t, rfl⟩
            /-
              case intro
              L : FirstOrder.Language
              M : Type w
              N : Type u_1
              P : Type u_2
              inst✝² : L.Structure M
              inst✝¹ : L.Structure N
              inst✝ : L.Structure P
              S : L.Substructure M
              s : Set (L.Substructure M)
              n : Nat
              f : L.Functions n
              t : L.Substructure M
              ⊢ FirstOrder.Language.ClosedUnder f ((fun t => Set.iInter fun h => ↑t) t)
            -/
            by_cases h : t ∈ s
              /-
                case pos
                L : FirstOrder.Language
                M : Type w
                N : Type u_1
                P : Type u_2
                inst✝² : L.Structure M
                inst✝¹ : L.Structure N
                inst✝ : L.Structure P
                S : L.Substructure M
                s : Set (L.Substructure M)
                n : Nat
                f : L.Functions n
                t : L.Substructure M
                h : Membership.mem s t
                ⊢ FirstOrder.Language.ClosedUnder f ((fun t => Set.iInter fun h => ↑t) t)
              -/
            · simpa [h] using t.fun_mem f
              /-
                🎉 no goals
              -/
              /-
                case neg
                L : FirstOrder.Language
                M : Type w
                N : Type u_1
                P : Type u_2
                inst✝² : L.Structure M
                inst✝¹ : L.Structure N
                inst✝ : L.Structure P
                S : L.Substructure M
                s : Set (L.Substructure M)
                n : Nat
                f : L.Functions n
                t : L.Substructure M
                h : Not (Membership.mem s t)
                ⊢ FirstOrder.Language.ClosedUnder f ((fun t => Set.iInter fun h => ↑t) t)
              -/
            · simp [h]) }⟩
              /-
                🎉 no goals
              -/


@[simp, norm_cast]
theorem coe_sInf (S : Set (L.Substructure M)) :
    ((sInf S : L.Substructure M) : Set M) = ⋂ s ∈ S, (s : Set M) :=
  rfl


theorem mem_sInf {S : Set (L.Substructure M)} {x : M} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p :=
  Set.mem_iInter₂


theorem mem_iInf {ι : Sort*} {S : ι → L.Substructure M} {x : M} :
                                        /-
                                          L : FirstOrder.Language
                                          M : Type w
                                          inst✝ : L.Structure M
                                          ι : Sort u_3
                                          S : ι → L.Substructure M
                                          x : M
                                          ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
                                        -/
    (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by simp only [iInf, mem_sInf, Set.forall_mem_range]
                                        /-
                                          🎉 no goals
                                        -/


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} {S : ι → L.Substructure M} :
    ((⨅ i, S i : L.Substructure M) : Set M) = ⋂ i, (S i : Set M) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ι : Sort u_3
    S : ι → L.Substructure M
    ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
  -/
  simp only [iInf, coe_sInf, Set.biInter_range]
  /-
    🎉 no goals
  -/


/-- Substructures of a structure form a complete lattice. -/
instance instCompleteLattice : CompleteLattice (L.Substructure M) :=
  { completeLatticeOfInf (L.Substructure M) fun _ =>
      IsGLB.of_image
        (fun {S T : L.Substructure M} => show (S : Set M) ≤ T ↔ S ≤ T from SetLike.coe_subset_coe)
        isGLB_biInf with
    le := (· ≤ ·)
    lt := (· < ·)
    top := ⊤
    le_top := fun _ x _ => mem_top x
    inf := (· ⊓ ·)
    sInf := InfSet.sInf
    le_inf := fun _a _b _c ha hb _x hx => ⟨ha hx, hb hx⟩
    inf_le_left := fun _ _ _ => And.left
    inf_le_right := fun _ _ _ => And.right }


/-- The `L.Substructure` generated by a set. -/
def closure : LowerAdjoint ((↑) : L.Substructure M → Set M) :=
  ⟨fun s => sInf { S | s ⊆ S }, fun _ _ =>
    ⟨Set.Subset.trans fun _x hx => mem_sInf.2 fun _S hS => hS hx, fun h => sInf_le h⟩⟩


theorem mem_closure {x : M} : x ∈ closure L s ↔ ∀ S : L.Substructure M, s ⊆ S → x ∈ S :=
  mem_sInf


/-- The substructure generated by a set includes the set. -/
@[simp]
theorem subset_closure : s ⊆ closure L s :=
  (closure L).le_closure s


theorem not_mem_of_not_mem_closure {P : M} (hP : P ∉ closure L s) : P ∉ s := fun h =>
  hP (subset_closure h)


@[simp]
theorem closed (S : L.Substructure M) : (closure L).closed (S : Set M) :=
  congr rfl ((closure L).eq_of_le Set.Subset.rfl fun _x xS => mem_closure.2 fun _T hT => hT xS)


/-- A substructure `S` includes `closure L s` if and only if it includes `s`. -/
@[simp]
theorem closure_le : closure L s ≤ S ↔ s ⊆ S :=
  (closure L).closure_le_closed_iff_le s S.closed


/-- Substructure closure of a set is monotone in its argument: if `s ⊆ t`,
then `closure L s ≤ closure L t`. -/
@[gcongr]
theorem closure_mono ⦃s t : Set M⦄ (h : s ⊆ t) : closure L s ≤ closure L t :=
  (closure L).monotone h


theorem closure_eq_of_le (h₁ : s ⊆ S) (h₂ : S ≤ closure L s) : closure L s = S :=
  (closure L).eq_of_le h₁ h₂


theorem coe_closure_eq_range_term_realize :
    (closure L s : Set M) = range (@Term.realize L _ _ _ ((↑) : s → M)) := by
  let S : L.Substructure M := ⟨range (Term.realize (L := L) ((↑) : s → M)), fun {n} f x hx => by
    simp only [mem_range] at *
    refine ⟨func f fun i => Classical.choose (hx i), ?_⟩
    simp only [Term.realize, fun i => Classical.choose_spec (hx i)]⟩
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    S : L.Substructure M := { carrier := Set.range (FirstOrder.Language.Term.reali …
    ⊢ Eq (↑((FirstOrder.Language.Substructure.closure L).toFun s)) (Set.range (Fir …
  -/
  change _ = (S : Set M)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    S : L.Substructure M := { carrier := Set.range (FirstOrder.Language.Term.reali …
    ⊢ Eq ↑((FirstOrder.Language.Substructure.closure L).toFun s) ↑S
  -/
  rw [← SetLike.ext'_iff]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    S : L.Substructure M := { carrier := Set.range (FirstOrder.Language.Term.reali …
    ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun s) S
  -/
  refine closure_eq_of_le (fun x hx => ⟨var ⟨x, hx⟩, rfl⟩) (le_sInf fun S' hS' => ?_)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    S : L.Substructure M := { carrier := Set.range (FirstOrder.Language.Term.reali …
    S' : L.Substructure M
    hS' : Membership.mem (setOf fun S => HasSubset.Subset s ↑S) S'
    ⊢ LE.le S S'
  -/
  rintro _ ⟨t, rfl⟩
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    S : L.Substructure M := { carrier := Set.range (FirstOrder.Language.Term.reali …
    S' : L.Substructure M
    hS' : Membership.mem (setOf fun S => HasSubset.Subset s ↑S) S'
    t : L.Term (Subtype fun x => Membership.mem s x)
    ⊢ Membership.mem S' (FirstOrder.Language.Term.realize Subtype.val t)
  -/
  exact t.realize_mem _ fun i => hS' i.2
  /-
    🎉 no goals
  -/


instance small_closure [Small.{u} s] : Small.{u} (closure L s) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝³ : L.Structure M
    inst✝² : L.Structure N
    inst✝¹ : L.Structure P
    S : L.Substructure M
    s : Set M
    inst✝ : Small.{u, w} ↑s
    ⊢ Small.{u, w} (Subtype fun x => Membership.mem ((FirstOrder.Language.Substruc …
  -/
  rw [← SetLike.coe_sort_coe, Substructure.coe_closure_eq_range_term_realize]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝³ : L.Structure M
    inst✝² : L.Structure N
    inst✝¹ : L.Structure P
    S : L.Substructure M
    s : Set M
    inst✝ : Small.{u, w} ↑s
    ⊢ Small.{u, w} ↑(Set.range (FirstOrder.Language.Term.realize Subtype.val))
  -/
  exact small_range _
  /-
    🎉 no goals
  -/


theorem mem_closure_iff_exists_term {x : M} :
    x ∈ closure L s ↔ ∃ t : L.Term s, t.realize ((↑) : s → M) = x := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    x : M
    ⊢ Iff (Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun s) x …
  -/
  rw [← SetLike.mem_coe, coe_closure_eq_range_term_realize, mem_range]
  /-
    🎉 no goals
  -/


theorem lift_card_closure_le_card_term : Cardinal.lift.{max u w} #(closure L s) ≤ #(L.Term s) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ LE.le (Cardinal.lift.{max u w, w} (Cardinal.mk (Subtype fun x => Membership. …
  -/
  rw [← SetLike.coe_sort_coe, coe_closure_eq_range_term_realize]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ LE.le (Cardinal.lift.{max u w, w} (Cardinal.mk ↑(Set.range (FirstOrder.Langu …
  -/
  rw [← Cardinal.lift_id'.{w, max u w} #(L.Term s)]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ LE.le (Cardinal.lift.{max u w, w} (Cardinal.mk ↑(Set.range (FirstOrder.Langu …
  -/
  exact Cardinal.mk_range_le_lift
  /-
    🎉 no goals
  -/


theorem lift_card_closure_le :
    Cardinal.lift.{u, w} #(closure L s) ≤
      max ℵ₀ (Cardinal.lift.{u, w} #s + Cardinal.lift.{w, u} #(Σi, L.Functions i)) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ LE.le (Cardinal.lift.{u, w} (Cardinal.mk (Subtype fun x => Membership.mem (( …
  -/
  rw [← lift_umax]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ LE.le (Cardinal.lift.{max w u, w} (Cardinal.mk (Subtype fun x => Membership. …
  -/
  refine lift_card_closure_le_card_term.trans (Term.card_le.trans ?_)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ LE.le (Max.max Cardinal.aleph0 (Cardinal.mk (Sum (↑s) (Sigma fun i => L.Func …
  -/
  rw [mk_sum, lift_umax.{w, u}]
  /-
    🎉 no goals
  -/


lemma mem_closed_iff (s : Set M) :
    s ∈ (closure L).closed ↔ ∀ {n}, ∀ f : L.Functions n, ClosedUnder f s := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    ⊢ Iff (Membership.mem (FirstOrder.Language.Substructure.closure L).closed s) ( …
  -/
  refine ⟨fun h n f => ?_, fun h => ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      s : Set M
      h : Membership.mem (FirstOrder.Language.Substructure.closure L).closed s
      n : Nat
      f : L.Functions n
      ⊢ FirstOrder.Language.ClosedUnder f s
    -/
  · rw [← h]
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      s : Set M
      h : Membership.mem (FirstOrder.Language.Substructure.closure L).closed s
      n : Nat
      f : L.Functions n
      ⊢ FirstOrder.Language.ClosedUnder f ↑((FirstOrder.Language.Substructure.closur …
    -/
    exact Substructure.fun_mem _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      s : Set M
      h : ∀ {n : Nat} (f : L.Functions n), FirstOrder.Language.ClosedUnder f s
      ⊢ Membership.mem (FirstOrder.Language.Substructure.closure L).closed s
    -/
  · have h' : closure L s = ⟨s, h⟩ := closure_eq_of_le (refl _) subset_closure
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      s : Set M
      h : ∀ {n : Nat} (f : L.Functions n), FirstOrder.Language.ClosedUnder f s
      h' : Eq ((FirstOrder.Language.Substructure.closure L).toFun s) { carrier := s, …
      ⊢ Membership.mem (FirstOrder.Language.Substructure.closure L).closed s
    -/
    exact congr_arg _ h'
    /-
      🎉 no goals
    -/


lemma mem_closed_of_isRelational [L.IsRelational] (s : Set M) : s ∈ (closure L).closed :=
  (mem_closed_iff s).2 isEmptyElim


@[simp]
lemma closure_eq_of_isRelational [L.IsRelational] (s : Set M) : closure L s = s :=
  LowerAdjoint.closure_eq_self_of_mem_closed _ (mem_closed_of_isRelational L s)


@[simp]
lemma mem_closure_iff_of_isRelational [L.IsRelational] (s : Set M) (m : M) :
    m ∈ closure L s ↔ m ∈ s := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    inst✝ : L.IsRelational
    s : Set M
    m : M
    ⊢ Iff (Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun s) m …
  -/
  rw [← SetLike.mem_coe, closure_eq_of_isRelational]
  /-
    🎉 no goals
  -/


theorem _root_.Set.Countable.substructure_closure
    [Countable (Σl, L.Functions l)] (h : s.Countable) : Countable.{w + 1} (closure L s) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    s : Set M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    h : s.Countable
    ⊢ Countable (Subtype fun x => Membership.mem ((FirstOrder.Language.Substructur …
  -/
  haveI : Countable s := h.to_subtype
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    s : Set M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    h : s.Countable
    this : Countable ↑s
    ⊢ Countable (Subtype fun x => Membership.mem ((FirstOrder.Language.Substructur …
  -/
  rw [← mk_le_aleph0_iff, ← lift_le_aleph0]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    s : Set M
    inst✝ : Countable (Sigma fun l => L.Functions l)
    h : s.Countable
    this : Countable ↑s
    ⊢ LE.le (Cardinal.lift.{?u.29173, w} (Cardinal.mk (Subtype fun x => Membership …
  -/
  exact lift_card_closure_le_card_term.trans mk_le_aleph0
  /-
    🎉 no goals
  -/


/-- An induction principle for closure membership. If `p` holds for all elements of `s`, and
is preserved under function symbols, then `p` holds for all elements of the closure of `s`. -/
@[elab_as_elim]
theorem closure_induction {p : M → Prop} {x} (h : x ∈ closure L s) (Hs : ∀ x ∈ s, p x)
    (Hfun : ∀ {n : ℕ} (f : L.Functions n), ClosedUnder f (setOf p)) : p x :=
  (@closure_le L M _ ⟨setOf p, fun {_} => Hfun⟩ _).2 Hs h


/-- If `s` is a dense set in a structure `M`, `Substructure.closure L s = ⊤`, then in order to prove
that some predicate `p` holds for all `x : M` it suffices to verify `p x` for `x ∈ s`, and verify
that `p` is preserved under function symbols. -/
@[elab_as_elim]
theorem dense_induction {p : M → Prop} (x : M) {s : Set M} (hs : closure L s = ⊤)
    (Hs : ∀ x ∈ s, p x) (Hfun : ∀ {n : ℕ} (f : L.Functions n), ClosedUnder f (setOf p)) : p x := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    p : M → Prop
    x : M
    s : Set M
    hs : Eq ((FirstOrder.Language.Substructure.closure L).toFun s) Top.top
    Hs : ∀ (x : M), Membership.mem s x → p x
    Hfun : ∀ {n : Nat} (f : L.Functions n), FirstOrder.Language.ClosedUnder f (set …
    ⊢ p x
  -/
  have : ∀ x ∈ closure L s, p x := fun x hx => closure_induction hx Hs fun {n} => Hfun
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    p : M → Prop
    x : M
    s : Set M
    hs : Eq ((FirstOrder.Language.Substructure.closure L).toFun s) Top.top
    Hs : ∀ (x : M), Membership.mem s x → p x
    Hfun : ∀ {n : Nat} (f : L.Functions n), FirstOrder.Language.ClosedUnder f (set …
    this : ∀ (x : M), Membership.mem ((FirstOrder.Language.Substructure.closure L) …
    ⊢ p x
  -/
  simpa [hs] using this x
  /-
    🎉 no goals
  -/


/-- `closure` forms a Galois insertion with the coercion to set. -/
protected def gi : GaloisInsertion (@closure L M _) (↑) where
  choice s _ := closure L s
  gc := (closure L).gc
  le_l_u _ := subset_closure
  choice_eq _ _ := rfl


/-- Closure of a substructure `S` equals `S`. -/
@[simp]
theorem closure_eq : closure L (S : Set M) = S :=
  (Substructure.gi L M).l_u_eq S


@[simp]
theorem closure_empty : closure L (∅ : Set M) = ⊥ :=
  (Substructure.gi L M).gc.l_bot


@[simp]
theorem closure_univ : closure L (univ : Set M) = ⊤ :=
  @coe_top L M _ ▸ closure_eq ⊤


theorem closure_union (s t : Set M) : closure L (s ∪ t) = closure L s ⊔ closure L t :=
  (Substructure.gi L M).gc.l_sup


theorem closure_iUnion {ι} (s : ι → Set M) : closure L (⋃ i, s i) = ⨆ i, closure L (s i) :=
  (Substructure.gi L M).gc.l_iSup


theorem closure_insert (s : Set M) (m : M) : closure L (insert m s) = closure L {m} ⊔ closure L s :=
  closure_union {m} s


instance small_bot : Small.{u} (⊥ : L.Substructure M) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : L.Structure P
    S : L.Substructure M
    s : Set M
    ⊢ Small.{u, w} (Subtype fun x => Membership.mem Bot.bot x)
  -/
  rw [← closure_empty]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : L.Structure P
    S : L.Substructure M
    s : Set M
    ⊢ Small.{u, w} (Subtype fun x => Membership.mem ((FirstOrder.Language.Substruc …
  -/
  haveI : Small.{u} (∅ : Set M) := small_subsingleton _
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : L.Structure P
    S : L.Substructure M
    s : Set M
    this : Small.{u, w} ↑EmptyCollection.emptyCollection
    ⊢ Small.{u, w} (Subtype fun x => Membership.mem ((FirstOrder.Language.Substruc …
  -/
  exact Substructure.small_closure
  /-
    🎉 no goals
  -/


theorem iSup_eq_closure {ι : Sort*} (S : ι → L.Substructure M) :
                                                    /-
                                                      L : FirstOrder.Language
                                                      M : Type w
                                                      inst✝ : L.Structure M
                                                      ι : Sort u_3
                                                      S : ι → L.Substructure M
                                                      ⊢ Eq (iSup fun i => S i) ((FirstOrder.Language.Substructure.closure L).toFun ( …
                                                    -/
    ⨆ i, S i = closure L (⋃ i, (S i : Set M)) := by simp_rw [closure_iUnion, closure_eq]
                                                    /-
                                                      🎉 no goals
                                                    -/

-- This proof uses the fact that `Substructure.closure` is finitary.

theorem mem_iSup_of_directed {ι : Type*} [hι : Nonempty ι] {S : ι → L.Substructure M}
    (hS : Directed (· ≤ ·) S) {x : M} :
    x ∈ ⨆ i, S i ↔ ∃ i, x ∈ S i := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ι : Type u_3
    hι : Nonempty ι
    S : ι → L.Substructure M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Iff (Membership.mem (iSup fun i => S i) x) (Exists fun i => Membership.mem ( …
  -/
  refine ⟨?_, fun ⟨i, hi⟩ ↦ le_iSup S i hi⟩
  suffices x ∈ closure L (⋃ i, (S i : Set M)) → ∃ i, x ∈ S i by
    simpa only [closure_iUnion, closure_eq (S _)] using this
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ι : Type u_3
    hι : Nonempty ι
    S : ι → L.Substructure M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun (Set.iUni …
  -/
  refine fun hx ↦ closure_induction hx (fun _ ↦ mem_iUnion.1) (fun f v hC ↦ ?_)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ι : Type u_3
    hι : Nonempty ι
    S : ι → L.Substructure M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    hx : Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun (Set.i …
    n✝ : Nat
    f : L.Functions n✝
    v : Fin n✝ → M
    hC : ∀ (i : Fin n✝), Membership.mem (setOf fun x => Exists fun i => Membership …
    ⊢ Membership.mem (setOf fun x => Exists fun i => Membership.mem (S i) x) (Firs …
  -/
  simp_rw [Set.mem_setOf] at *
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ι : Type u_3
    hι : Nonempty ι
    S : ι → L.Substructure M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    hx : Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun (Set.i …
    n✝ : Nat
    f : L.Functions n✝
    v : Fin n✝ → M
    hC : ∀ (i : Fin n✝), Exists fun i_1 => Membership.mem (S i_1) (v i)
    ⊢ Exists fun i => Membership.mem (S i) (FirstOrder.Language.Structure.funMap f …
  -/
  have ⟨i, hi⟩ := hS.finite_le (fun i ↦ Classical.choose (hC i))
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ι : Type u_3
    hι : Nonempty ι
    S : ι → L.Substructure M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    hx : Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun (Set.i …
    n✝ : Nat
    f : L.Functions n✝
    v : Fin n✝ → M
    hC : ∀ (i : Fin n✝), Exists fun i_1 => Membership.mem (S i_1) (v i)
    i : ι
    hi : ∀ (i_1 : Fin n✝), LE.le (S (Classical.choose ⋯)) (S i)
    ⊢ Exists fun i => Membership.mem (S i) (FirstOrder.Language.Structure.funMap f …
  -/
  refine ⟨i, (S i).fun_mem f v (fun j ↦ hi j (Classical.choose_spec (hC j)))⟩
  /-
    🎉 no goals
  -/

-- This proof uses the fact that `Substructure.closure` is finitary.

theorem mem_sSup_of_directedOn {S : Set (L.Substructure M)} (Sne : S.Nonempty)
    (hS : DirectedOn (· ≤ ·) S) {x : M} :
    x ∈ sSup S ↔ ∃ s ∈ S, x ∈ s := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : Set (L.Substructure M)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  haveI : Nonempty S := Sne.to_subtype
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : Set (L.Substructure M)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : M
    this : Nonempty ↑S
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  simp only [sSup_eq_iSup', mem_iSup_of_directed hS.directed_val, Subtype.exists, exists_prop]
  /-
    🎉 no goals
  -/


instance [IsEmpty L.Constants] : IsEmpty (⊥ : L.Substructure M) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝³ : L.Structure M
    inst✝² : L.Structure N
    inst✝¹ : L.Structure P
    S : L.Substructure M
    s : Set M
    inst✝ : IsEmpty L.Constants
    ⊢ IsEmpty (Subtype fun x => Membership.mem Bot.bot x)
  -/
  refine (isEmpty_subtype _).2 (fun x => ?_)
  have h : (∅ : Set M) ∈ (closure L).closed := by
    rw [mem_closed_iff]
    intro n f
    cases n
    · exact isEmptyElim f
    · intro x hx
      simp only [mem_empty_iff_false, forall_const] at hx
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝³ : L.Structure M
    inst✝² : L.Structure N
    inst✝¹ : L.Structure P
    S : L.Substructure M
    s : Set M
    inst✝ : IsEmpty L.Constants
    x : M
    h : Membership.mem (FirstOrder.Language.Substructure.closure L).closed EmptyCo …
    ⊢ Not (Membership.mem Bot.bot x)
  -/
  rw [← closure_empty, ← SetLike.mem_coe, h]
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    P : Type u_2
    inst✝³ : L.Structure M
    inst✝² : L.Structure N
    inst✝¹ : L.Structure P
    S : L.Substructure M
    s : Set M
    inst✝ : IsEmpty L.Constants
    x : M
    h : Membership.mem (FirstOrder.Language.Substructure.closure L).closed EmptyCo …
    ⊢ Not (Membership.mem EmptyCollection.emptyCollection x)
  -/
  exact Set.not_mem_empty _
  /-
    🎉 no goals
  -/


/-- The preimage of a substructure along a homomorphism is a substructure. -/
@[simps]
def comap (φ : M →[L] N) (S : L.Substructure N) : L.Substructure M where
  carrier := φ ⁻¹' S
  fun_mem {n} f x hx := by
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S✝ : L.Substructure M
      s : Set M
      φ : L.Hom M N
      S : L.Substructure N
      n : Nat
      f : L.Functions n
      x : Fin n → M
      hx : ∀ (i : Fin n), Membership.mem (Set.preimage ⇑φ ↑S) (x i)
      ⊢ Membership.mem (Set.preimage ⇑φ ↑S) (FirstOrder.Language.Structure.funMap f x)
    -/
    rw [mem_preimage, φ.map_fun]
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S✝ : L.Substructure M
      s : Set M
      φ : L.Hom M N
      S : L.Substructure N
      n : Nat
      f : L.Functions n
      x : Fin n → M
      hx : ∀ (i : Fin n), Membership.mem (Set.preimage ⇑φ ↑S) (x i)
      ⊢ Membership.mem (↑S) (FirstOrder.Language.Structure.funMap f (Function.comp ( …
    -/
    exact S.fun_mem f (φ ∘ x) hx
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_comap {S : L.Substructure N} {f : M →[L] N} {x : M} : x ∈ S.comap f ↔ f x ∈ S :=
  Iff.rfl


theorem comap_comap (S : L.Substructure P) (g : N →[L] P) (f : M →[L] N) :
    (S.comap g).comap f = S.comap (g.comp f) :=
  rfl


@[simp]
theorem comap_id (S : L.Substructure P) : S.comap (Hom.id _ _) = S :=
          /-
            L : FirstOrder.Language
            P : Type u_2
            inst✝ : L.Structure P
            S : L.Substructure P
            ⊢ ∀ (x : P), Iff (Membership.mem (FirstOrder.Language.Substructure.comap (Firs …
          -/
  ext (by simp)
          /-
            🎉 no goals
          -/


/-- The image of a substructure along a homomorphism is a substructure. -/
@[simps]
def map (φ : M →[L] N) (S : L.Substructure M) : L.Substructure N where
  carrier := φ '' S
  fun_mem {n} f x hx :=
    (mem_image _ _ _).1
      ⟨funMap f fun i => Classical.choose (hx i),
        S.fun_mem f _ fun i => (Classical.choose_spec (hx i)).1, by
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝² : L.Structure M
          inst✝¹ : L.Structure N
          inst✝ : L.Structure P
          S✝ : L.Substructure M
          s : Set M
          φ : L.Hom M N
          S : L.Substructure M
          n : Nat
          f : L.Functions n
          x : Fin n → N
          hx : ∀ (i : Fin n), Membership.mem (Set.image ⇑φ ↑S) (x i)
          ⊢ Eq (φ (FirstOrder.Language.Structure.funMap f fun i => Classical.choose ⋯))  …
        -/
        simp only [Hom.map_fun, SetLike.mem_coe]
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝² : L.Structure M
          inst✝¹ : L.Structure N
          inst✝ : L.Structure P
          S✝ : L.Substructure M
          s : Set M
          φ : L.Hom M N
          S : L.Substructure M
          n : Nat
          f : L.Functions n
          x : Fin n → N
          hx : ∀ (i : Fin n), Membership.mem (Set.image ⇑φ ↑S) (x i)
          ⊢ Eq (FirstOrder.Language.Structure.funMap f (Function.comp ⇑φ fun i => Classi …
        -/
        exact congr rfl (funext fun i => (Classical.choose_spec (hx i)).2)⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem mem_map {f : M →[L] N} {S : L.Substructure M} {y : N} :
    y ∈ S.map f ↔ ∃ x ∈ S, f x = y :=
  Iff.rfl


theorem mem_map_of_mem (f : M →[L] N) {S : L.Substructure M} {x : M} (hx : x ∈ S) : f x ∈ S.map f :=
  mem_image_of_mem f hx


theorem apply_coe_mem_map (f : M →[L] N) (S : L.Substructure M) (x : S) : f x ∈ S.map f :=
  mem_map_of_mem f x.prop


theorem map_map (g : N →[L] P) (f : M →[L] N) : (S.map f).map g = S.map (g.comp f) :=
  SetLike.coe_injective <| image_image _ _ _


theorem map_le_iff_le_comap {f : M →[L] N} {S : L.Substructure M} {T : L.Substructure N} :
    S.map f ≤ T ↔ S ≤ T.comap f :=
  image_subset_iff


theorem gc_map_comap (f : M →[L] N) : GaloisConnection (map f) (comap f) := fun _ _ =>
  map_le_iff_le_comap


theorem map_le_of_le_comap {T : L.Substructure N} {f : M →[L] N} : S ≤ T.comap f → S.map f ≤ T :=
  (gc_map_comap f).l_le


theorem le_comap_of_map_le {T : L.Substructure N} {f : M →[L] N} : S.map f ≤ T → S ≤ T.comap f :=
  (gc_map_comap f).le_u


theorem le_comap_map {f : M →[L] N} : S ≤ (S.map f).comap f :=
  (gc_map_comap f).le_u_l _


theorem map_comap_le {S : L.Substructure N} {f : M →[L] N} : (S.comap f).map f ≤ S :=
  (gc_map_comap f).l_u_le _


theorem monotone_map {f : M →[L] N} : Monotone (map f) :=
  (gc_map_comap f).monotone_l


theorem monotone_comap {f : M →[L] N} : Monotone (comap f) :=
  (gc_map_comap f).monotone_u


@[simp]
theorem map_comap_map {f : M →[L] N} : ((S.map f).comap f).map f = S.map f :=
  (gc_map_comap f).l_u_l_eq_l _


@[simp]
theorem comap_map_comap {S : L.Substructure N} {f : M →[L] N} :
    ((S.comap f).map f).comap f = S.comap f :=
  (gc_map_comap f).u_l_u_eq_u _


theorem map_sup (S T : L.Substructure M) (f : M →[L] N) : (S ⊔ T).map f = S.map f ⊔ T.map f :=
  (gc_map_comap f).l_sup


theorem map_iSup {ι : Sort*} (f : M →[L] N) (s : ι → L.Substructure M) :
    (⨆ i, s i).map f = ⨆ i, (s i).map f :=
  (gc_map_comap f).l_iSup


theorem comap_inf (S T : L.Substructure N) (f : M →[L] N) :
    (S ⊓ T).comap f = S.comap f ⊓ T.comap f :=
  (gc_map_comap f).u_inf


theorem comap_iInf {ι : Sort*} (f : M →[L] N) (s : ι → L.Substructure N) :
    (⨅ i, s i).comap f = ⨅ i, (s i).comap f :=
  (gc_map_comap f).u_iInf


@[simp]
theorem map_bot (f : M →[L] N) : (⊥ : L.Substructure M).map f = ⊥ :=
  (gc_map_comap f).l_bot


@[simp]
theorem comap_top (f : M →[L] N) : (⊤ : L.Substructure N).comap f = ⊤ :=
  (gc_map_comap f).u_top


@[simp]
theorem map_id (S : L.Substructure M) : S.map (Hom.id L M) = S :=
  SetLike.coe_injective <| Set.image_id _


theorem map_closure (f : M →[L] N) (s : Set M) : (closure L s).map f = closure L (f '' s) :=
  Eq.symm <|
    closure_eq_of_le (Set.image_subset f subset_closure) <|
      map_le_iff_le_comap.2 <| closure_le.2 fun x hx => subset_closure ⟨x, hx, rfl⟩


@[simp]
theorem closure_image (f : M →[L] N) : closure L (f '' s) = map f (closure L s) :=
  (map_closure f s).symm


/-- `map f` and `comap f` form a `GaloisCoinsertion` when `f` is injective. -/
def gciMapComap (hf : Function.Injective f) : GaloisCoinsertion (map f) (comap f) :=
                                                     /-
                                                       L : FirstOrder.Language
                                                       M : Type w
                                                       N : Type u_1
                                                       P : Type u_2
                                                       inst✝² : L.Structure M
                                                       inst✝¹ : L.Structure N
                                                       inst✝ : L.Structure P
                                                       S✝ : L.Substructure M
                                                       s : Set M
                                                       ι : Type u_3
                                                       f : L.Hom M N
                                                       hf : Function.Injective ⇑f
                                                       S : L.Substructure M
                                                       x : M
                                                       ⊢ Membership.mem (FirstOrder.Language.Substructure.comap f (FirstOrder.Languag …
                                                     -/
  (gc_map_comap f).toGaloisCoinsertion fun S x => by simp [mem_comap, mem_map, hf.eq_iff]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem comap_map_eq_of_injective (S : L.Substructure M) : (S.map f).comap f = S :=
  (gciMapComap hf).u_l_eq _


theorem comap_surjective_of_injective : Function.Surjective (comap f) :=
  (gciMapComap hf).u_surjective


theorem map_injective_of_injective : Function.Injective (map f) :=
  (gciMapComap hf).l_injective


theorem comap_inf_map_of_injective (S T : L.Substructure M) : (S.map f ⊓ T.map f).comap f = S ⊓ T :=
  (gciMapComap hf).u_inf_l _ _


theorem comap_iInf_map_of_injective (S : ι → L.Substructure M) :
    (⨅ i, (S i).map f).comap f = ⨅ i, S i :=
  (gciMapComap hf).u_iInf_l _


theorem comap_sup_map_of_injective (S T : L.Substructure M) : (S.map f ⊔ T.map f).comap f = S ⊔ T :=
  (gciMapComap hf).u_sup_l _ _


theorem comap_iSup_map_of_injective (S : ι → L.Substructure M) :
    (⨆ i, (S i).map f).comap f = ⨆ i, S i :=
  (gciMapComap hf).u_iSup_l _


theorem map_le_map_iff_of_injective {S T : L.Substructure M} : S.map f ≤ T.map f ↔ S ≤ T :=
  (gciMapComap hf).l_le_l_iff


theorem map_strictMono_of_injective : StrictMono (map f) :=
  (gciMapComap hf).strictMono_l


/-- `map f` and `comap f` form a `GaloisInsertion` when `f` is surjective. -/
def giMapComap : GaloisInsertion (map f) (comap f) :=
  (gc_map_comap f).toGaloisInsertion fun S x h =>
    let ⟨y, hy⟩ := hf x
                     /-
                       L : FirstOrder.Language
                       M : Type w
                       N : Type u_1
                       P : Type u_2
                       inst✝² : L.Structure M
                       inst✝¹ : L.Structure N
                       inst✝ : L.Structure P
                       S✝ : L.Substructure M
                       s : Set M
                       ι : Type u_3
                       f : L.Hom M N
                       hf : Function.Surjective ⇑f
                       S : L.Substructure N
                       x : N
                       h : Membership.mem S x
                       y : M
                       hy : Eq (f y) x
                       ⊢ And (Membership.mem (FirstOrder.Language.Substructure.comap f S) y) (Eq (f y …
                     -/
    mem_map.2 ⟨y, by simp [hy, h]⟩
                     /-
                       🎉 no goals
                     -/


theorem map_comap_eq_of_surjective (S : L.Substructure N) : (S.comap f).map f = S :=
  (giMapComap hf).l_u_eq _


theorem map_surjective_of_surjective : Function.Surjective (map f) :=
  (giMapComap hf).l_surjective


theorem comap_injective_of_surjective : Function.Injective (comap f) :=
  (giMapComap hf).u_injective


theorem map_inf_comap_of_surjective (S T : L.Substructure N) :
    (S.comap f ⊓ T.comap f).map f = S ⊓ T :=
  (giMapComap hf).l_inf_u _ _


theorem map_iInf_comap_of_surjective (S : ι → L.Substructure N) :
    (⨅ i, (S i).comap f).map f = ⨅ i, S i :=
  (giMapComap hf).l_iInf_u _


theorem map_sup_comap_of_surjective (S T : L.Substructure N) :
    (S.comap f ⊔ T.comap f).map f = S ⊔ T :=
  (giMapComap hf).l_sup_u _ _


theorem map_iSup_comap_of_surjective (S : ι → L.Substructure N) :
    (⨆ i, (S i).comap f).map f = ⨆ i, S i :=
  (giMapComap hf).l_iSup_u _


theorem comap_le_comap_iff_of_surjective {S T : L.Substructure N} : S.comap f ≤ T.comap f ↔ S ≤ T :=
  (giMapComap hf).u_le_u_iff


theorem comap_strictMono_of_surjective : StrictMono (comap f) :=
  (giMapComap hf).strictMono_u


instance inducedStructure {S : L.Substructure M} : L.Structure S where
  funMap {_} f x := ⟨funMap f fun i => x i, S.fun_mem f (fun i => x i) fun i => (x i).2⟩
  RelMap {_} r x := RelMap r fun i => (x i : M)


/-- The natural embedding of an `L.Substructure` of `M` into `M`. -/
def subtype (S : L.Substructure M) : S ↪[L] M where
  toFun := (↑)
  inj' := Subtype.coe_injective


@[simp]
theorem coeSubtype : ⇑S.subtype = ((↑) : S → M) :=
  rfl


/-- The equivalence between the maximal substructure of a structure and the structure itself. -/
def topEquiv : (⊤ : L.Substructure M) ≃[L] M where
  toFun := subtype ⊤
  invFun m := ⟨m, mem_top m⟩
                   /-
                     L : FirstOrder.Language
                     M : Type w
                     N : Type u_1
                     P : Type u_2
                     inst✝² : L.Structure M
                     inst✝¹ : L.Structure N
                     inst✝ : L.Structure P
                     S : L.Substructure M
                     s : Set M
                     m : Subtype fun x => Membership.mem Top.top x
                     ⊢ Eq ((fun m => ⟨m, ⋯⟩) (Top.top.subtype m)) m
                   -/
  left_inv m := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv _ := rfl


@[simp]
theorem coe_topEquiv :
    ⇑(topEquiv : (⊤ : L.Substructure M) ≃[L] M) = ((↑) : (⊤ : L.Substructure M) → M) :=
  rfl


@[simp]
theorem realize_boundedFormula_top {α : Type*} {n : ℕ} {φ : L.BoundedFormula α n}
    {v : α → (⊤ : L.Substructure M)} {xs : Fin n → (⊤ : L.Substructure M)} :
    φ.Realize v xs ↔ φ.Realize (((↑) : _ → M) ∘ v) ((↑) ∘ xs) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u_3
    n : Nat
    φ : L.BoundedFormula α n
    v : α → Subtype fun x => Membership.mem Top.top x
    xs : Fin n → Subtype fun x => Membership.mem Top.top x
    ⊢ Iff (φ.Realize v xs) (φ.Realize (Function.comp Subtype.val v) (Function.comp …
  -/
  rw [← StrongHomClass.realize_boundedFormula Substructure.topEquiv φ]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u_3
    n : Nat
    φ : L.BoundedFormula α n
    v : α → Subtype fun x => Membership.mem Top.top x
    xs : Fin n → Subtype fun x => Membership.mem Top.top x
    ⊢ Iff (φ.Realize (Function.comp (⇑FirstOrder.Language.Substructure.topEquiv) v …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_formula_top {α : Type*} {φ : L.Formula α} {v : α → (⊤ : L.Substructure M)} :
    φ.Realize v ↔ φ.Realize (((↑) : (⊤ : L.Substructure M) → M) ∘ v) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u_3
    φ : L.Formula α
    v : α → Subtype fun x => Membership.mem Top.top x
    ⊢ Iff (φ.Realize v) (φ.Realize (Function.comp Subtype.val v))
  -/
  rw [← StrongHomClass.realize_formula Substructure.topEquiv φ]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u_3
    φ : L.Formula α
    v : α → Subtype fun x => Membership.mem Top.top x
    ⊢ Iff (φ.Realize (Function.comp (⇑FirstOrder.Language.Substructure.topEquiv) v …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A dependent version of `Substructure.closure_induction`. -/
@[elab_as_elim]
theorem closure_induction' (s : Set M) {p : ∀ x, x ∈ closure L s → Prop}
    (Hs : ∀ (x) (h : x ∈ s), p x (subset_closure h))
    (Hfun : ∀ {n : ℕ} (f : L.Functions n), ClosedUnder f { x | ∃ hx, p x hx }) {x}
    (hx : x ∈ closure L s) : p x hx := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    p : (x : M) → Membership.mem ((FirstOrder.Language.Substructure.closure L).toF …
    Hs : ∀ (x : M) (h : Membership.mem s x), p x ⋯
    Hfun : ∀ {n : Nat} (f : L.Functions n), FirstOrder.Language.ClosedUnder f (set …
    x : M
    hx : Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun s) x
    ⊢ p x hx
  -/
  refine Exists.elim ?_ fun (hx : x ∈ closure L s) (hc : p x hx) => hc
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    s : Set M
    p : (x : M) → Membership.mem ((FirstOrder.Language.Substructure.closure L).toF …
    Hs : ∀ (x : M) (h : Membership.mem s x), p x ⋯
    Hfun : ∀ {n : Nat} (f : L.Functions n), FirstOrder.Language.ClosedUnder f (set …
    x : M
    hx : Membership.mem ((FirstOrder.Language.Substructure.closure L).toFun s) x
    ⊢ Exists fun x_1 => p x x_1
  -/
  exact closure_induction hx (fun x hx => ⟨subset_closure hx, Hs x hx⟩) @Hfun
  /-
    🎉 no goals
  -/


/-- Reduces the language of a substructure along a language hom. -/
def substructureReduct (φ : L →ᴸ L') [φ.IsExpansionOn M] :
    L'.Substructure M ↪o L.Substructure M where
  toFun S :=
    { carrier := S
      fun_mem := fun {n} f x hx => by
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝⁴ : L.Structure M
          inst✝³ : L.Structure N
          inst✝² : L.Structure P
          S✝ : L.Substructure M
          L' : FirstOrder.Language
          inst✝¹ : L'.Structure M
          φ : L.LHom L'
          inst✝ : φ.IsExpansionOn M
          S : L'.Substructure M
          n : Nat
          f : L.Functions n
          x : Fin n → M
          hx : ∀ (i : Fin n), Membership.mem (↑S) (x i)
          ⊢ Membership.mem (↑S) (FirstOrder.Language.Structure.funMap f x)
        -/
        have h := S.fun_mem (φ.onFunction f) x hx
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝⁴ : L.Structure M
          inst✝³ : L.Structure N
          inst✝² : L.Structure P
          S✝ : L.Substructure M
          L' : FirstOrder.Language
          inst✝¹ : L'.Structure M
          φ : L.LHom L'
          inst✝ : φ.IsExpansionOn M
          S : L'.Substructure M
          n : Nat
          f : L.Functions n
          x : Fin n → M
          hx : ∀ (i : Fin n), Membership.mem (↑S) (x i)
          h : Membership.mem (↑S) (FirstOrder.Language.Structure.funMap (φ.onFunction f) …
          ⊢ Membership.mem (↑S) (FirstOrder.Language.Structure.funMap f x)
        -/
        simp only [LHom.map_onFunction, Substructure.mem_carrier] at h
        /-
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝⁴ : L.Structure M
          inst✝³ : L.Structure N
          inst✝² : L.Structure P
          S✝ : L.Substructure M
          L' : FirstOrder.Language
          inst✝¹ : L'.Structure M
          φ : L.LHom L'
          inst✝ : φ.IsExpansionOn M
          S : L'.Substructure M
          n : Nat
          f : L.Functions n
          x : Fin n → M
          hx : ∀ (i : Fin n), Membership.mem (↑S) (x i)
          h : Membership.mem S (FirstOrder.Language.Structure.funMap f x)
          ⊢ Membership.mem (↑S) (FirstOrder.Language.Structure.funMap f x)
        -/
        exact h }
        /-
          🎉 no goals
        -/
  inj' S T h := by
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝⁴ : L.Structure M
      inst✝³ : L.Structure N
      inst✝² : L.Structure P
      S✝ : L.Substructure M
      L' : FirstOrder.Language
      inst✝¹ : L'.Structure M
      φ : L.LHom L'
      inst✝ : φ.IsExpansionOn M
      S T : L'.Substructure M
      h : Eq ((fun S => { carrier := ↑S, fun_mem := ⋯ }) S) ((fun S => { carrier :=  …
      ⊢ Eq S T
    -/
    simp only [SetLike.coe_set_eq, Substructure.mk.injEq] at h
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝⁴ : L.Structure M
      inst✝³ : L.Structure N
      inst✝² : L.Structure P
      S✝ : L.Substructure M
      L' : FirstOrder.Language
      inst✝¹ : L'.Structure M
      φ : L.LHom L'
      inst✝ : φ.IsExpansionOn M
      S T : L'.Substructure M
      h : Eq S T
      ⊢ Eq S T
    -/
    exact h
    /-
      🎉 no goals
    -/
  map_rel_iff' {_ _} := Iff.rfl


@[simp]
theorem mem_substructureReduct {x : M} {S : L'.Substructure M} :
    x ∈ φ.substructureReduct S ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_substructureReduct {S : L'.Substructure M} : (φ.substructureReduct S : Set M) = ↑S :=
  rfl


/-- Turns any substructure containing a constant set `A` into a `L[[A]]`-substructure. -/
def withConstants (S : L.Substructure M) {A : Set M} (h : A ⊆ S) : L[[A]].Substructure M where
  carrier := S
  fun_mem {n} f := by
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S✝ S : L.Substructure M
      A : Set M
      h : HasSubset.Subset A ↑S
      n : Nat
      f : (L.withConstants ↑A).Functions n
      ⊢ FirstOrder.Language.ClosedUnder f ↑S
    -/
    cases' f with f f
      /-
        case inl
        L : FirstOrder.Language
        M : Type w
        N : Type u_1
        P : Type u_2
        inst✝² : L.Structure M
        inst✝¹ : L.Structure N
        inst✝ : L.Structure P
        S✝ S : L.Substructure M
        A : Set M
        h : HasSubset.Subset A ↑S
        n : Nat
        f : L.Functions n
        ⊢ FirstOrder.Language.ClosedUnder (Sum.inl f) ↑S
      -/
    · exact S.fun_mem f
      /-
        🎉 no goals
      -/
      /-
        case inr
        L : FirstOrder.Language
        M : Type w
        N : Type u_1
        P : Type u_2
        inst✝² : L.Structure M
        inst✝¹ : L.Structure N
        inst✝ : L.Structure P
        S✝ S : L.Substructure M
        A : Set M
        h : HasSubset.Subset A ↑S
        n : Nat
        f : (FirstOrder.Language.constantsOn ↑A).Functions n
        ⊢ FirstOrder.Language.ClosedUnder (Sum.inr f) ↑S
      -/
    · cases n
        /-
          case inr.zero
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝² : L.Structure M
          inst✝¹ : L.Structure N
          inst✝ : L.Structure P
          S✝ S : L.Substructure M
          A : Set M
          h : HasSubset.Subset A ↑S
          f : (FirstOrder.Language.constantsOn ↑A).Functions 0
          ⊢ FirstOrder.Language.ClosedUnder (Sum.inr f) ↑S
        -/
      · exact fun _ _ => h f.2
        /-
          🎉 no goals
        -/
        /-
          case inr.succ
          L : FirstOrder.Language
          M : Type w
          N : Type u_1
          P : Type u_2
          inst✝² : L.Structure M
          inst✝¹ : L.Structure N
          inst✝ : L.Structure P
          S✝ S : L.Substructure M
          A : Set M
          h : HasSubset.Subset A ↑S
          n✝ : Nat
          f : (FirstOrder.Language.constantsOn ↑A).Functions (HAdd.hAdd n✝ 1)
          ⊢ FirstOrder.Language.ClosedUnder (Sum.inr f) ↑S
        -/
      · exact isEmptyElim f
        /-
          🎉 no goals
        -/


@[simp]
theorem mem_withConstants {x : M} : x ∈ S.withConstants h ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_withConstants : (S.withConstants h : Set M) = ↑S :=
  rfl


@[simp]
theorem reduct_withConstants :
    (L.lhomWithConstants A).substructureReduct (S.withConstants h) = S := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    A : Set M
    h : HasSubset.Subset A ↑S
    ⊢ Eq ((L.lhomWithConstants ↑A).substructureReduct (S.withConstants h)) S
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    A : Set M
    h : HasSubset.Subset A ↑S
    x✝ : M
    ⊢ Iff (Membership.mem ((L.lhomWithConstants ↑A).substructureReduct (S.withCons …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem subset_closure_withConstants : A ⊆ closure (L[[A]]) s := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    ⊢ HasSubset.Subset A ↑((FirstOrder.Language.Substructure.closure (L.withConsta …
  -/
  intro a ha
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    a : M
    ha : Membership.mem A a
    ⊢ Membership.mem (↑((FirstOrder.Language.Substructure.closure (L.withConstants …
  -/
  simp only [SetLike.mem_coe]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    a : M
    ha : Membership.mem A a
    ⊢ Membership.mem ((FirstOrder.Language.Substructure.closure (L.withConstants ↑ …
  -/
  let a' : L[[A]].Constants := Sum.inr ⟨a, ha⟩
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    a : M
    ha : Membership.mem A a
    a' : (L.withConstants ↑A).Constants := Sum.inr ⟨a, ha⟩
    ⊢ Membership.mem ((FirstOrder.Language.Substructure.closure (L.withConstants ↑ …
  -/
  exact constants_mem a'
  /-
    🎉 no goals
  -/


theorem closure_withConstants_eq :
    closure (L[[A]]) s =
      (closure L (A ∪ s)).withConstants ((A.subset_union_left).trans subset_closure) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    ⊢ Eq ((FirstOrder.Language.Substructure.closure (L.withConstants ↑A)).toFun s) …
  -/
  refine closure_eq_of_le ((A.subset_union_right).trans subset_closure) ?_
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    ⊢ LE.le (((FirstOrder.Language.Substructure.closure L).toFun (Union.union A s) …
  -/
  rw [← (L.lhomWithConstants A).substructureReduct.le_iff_le]
  simp only [subset_closure, reduct_withConstants, closure_le, LHom.coe_substructureReduct,
    Set.union_subset_iff, and_true]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    A s : Set M
    ⊢ HasSubset.Subset A ↑((FirstOrder.Language.Substructure.closure (L.withConsta …
  -/
  exact subset_closure_withConstants
  /-
    🎉 no goals
  -/


/-- The restriction of a first-order hom to a substructure `s ⊆ M` gives a hom `s → N`. -/
@[simps!]
def domRestrict (f : M →[L] N) (p : L.Substructure M) : p →[L] N :=
  f.comp p.subtype.toHom


/-- A first-order hom `f : M → N` whose values lie in a substructure `p ⊆ N` can be restricted to a
hom `M → p`. -/
@[simps]
def codRestrict (p : L.Substructure N) (f : M →[L] N) (h : ∀ c, f c ∈ p) : M →[L] p where
  toFun c := ⟨f c, h c⟩
                         /-
                           L : FirstOrder.Language
                           M : Type w
                           N : Type u_1
                           P : Type u_2
                           inst✝² : L.Structure M
                           inst✝¹ : L.Structure N
                           inst✝ : L.Structure P
                           S : L.Substructure M
                           p : L.Substructure N
                           f✝ : L.Hom M N
                           h : ∀ (c : M), Membership.mem p (f✝ c)
                           n : Nat
                           f : L.Functions n
                           x : Fin n → M
                           ⊢ Eq ((fun c => ⟨f✝ c, ⋯⟩) (FirstOrder.Language.Structure.funMap f x)) (FirstO …
                         -/
  map_fun' {n} f x := by aesop
                         /-
                           🎉 no goals
                         -/
  map_rel' {_} R x h := f.map_rel R x h


@[simp]
theorem comp_codRestrict (f : M →[L] N) (g : N →[L] P) (p : L.Substructure P) (h : ∀ b, g b ∈ p) :
    ((codRestrict p g h).comp f : M →[L] p) = codRestrict p (g.comp f) fun _ => h _ :=
  ext fun _ => rfl


@[simp]
theorem subtype_comp_codRestrict (f : M →[L] N) (p : L.Substructure N) (h : ∀ b, f b ∈ p) :
    p.subtype.toHom.comp (codRestrict p f h) = f :=
  ext fun _ => rfl


/-- The range of a first-order hom `f : M → N` is a submodule of `N`.
See Note [range copy pattern]. -/
def range (f : M →[L] N) : L.Substructure N :=
  (map f ⊤).copy (Set.range f) Set.image_univ.symm


theorem range_coe (f : M →[L] N) : (range f : Set N) = Set.range f :=
  rfl


@[simp]
theorem mem_range {f : M →[L] N} {x} : x ∈ range f ↔ ∃ y, f y = x :=
  Iff.rfl


theorem range_eq_map (f : M →[L] N) : f.range = map f ⊤ := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Hom M N
    ⊢ Eq f.range (FirstOrder.Language.Substructure.map f Top.top)
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Hom M N
    x✝ : N
    ⊢ Iff (Membership.mem f.range x✝) (Membership.mem (FirstOrder.Language.Substru …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_range_self (f : M →[L] N) (x : M) : f x ∈ f.range :=
  ⟨x, rfl⟩


@[simp]
theorem range_id : range (id L M) = ⊤ :=
  SetLike.coe_injective Set.range_id


theorem range_comp (f : M →[L] N) (g : N →[L] P) : range (g.comp f : M →[L] P) = map g (range f) :=
  SetLike.coe_injective (Set.range_comp g f)


theorem range_comp_le_range (f : M →[L] N) (g : N →[L] P) : range (g.comp f : M →[L] P) ≤ range g :=
  SetLike.coe_mono (Set.range_comp_subset_range f g)


theorem range_eq_top {f : M →[L] N} : range f = ⊤ ↔ Function.Surjective f := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Hom M N
    ⊢ Iff (Eq f.range Top.top) (Function.Surjective ⇑f)
  -/
  rw [SetLike.ext'_iff, range_coe, coe_top, Set.range_eq_univ]
  /-
    🎉 no goals
  -/


theorem range_le_iff_comap {f : M →[L] N} {p : L.Substructure N} : range f ≤ p ↔ comap f p = ⊤ := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Hom M N
    p : L.Substructure N
    ⊢ Iff (LE.le f.range p) (Eq (FirstOrder.Language.Substructure.comap f p) Top.t …
  -/
  rw [range_eq_map, map_le_iff_le_comap, eq_top_iff]
  /-
    🎉 no goals
  -/


theorem map_le_range {f : M →[L] N} {p : L.Substructure M} : map f p ≤ range f :=
  SetLike.coe_mono (Set.image_subset_range f p)


/-- The substructure of elements `x : M` such that `f x = g x` -/
def eqLocus (f g : M →[L] N) : Substructure L M where
  carrier := { x : M | f x = g x }
  fun_mem {n} fn x hx := by
    have h : f ∘ x = g ∘ x := by
      ext
      repeat' rw [Function.comp_apply]
      apply hx
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S : L.Substructure M
      f g : L.Hom M N
      n : Nat
      fn : L.Functions n
      x : Fin n → M
      hx : ∀ (i : Fin n), Membership.mem (setOf fun x => Eq (f x) (g x)) (x i)
      h : Eq (Function.comp (⇑f) x) (Function.comp (⇑g) x)
      ⊢ Membership.mem (setOf fun x => Eq (f x) (g x)) (FirstOrder.Language.Structur …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


/-- If two `L.Hom`s are equal on a set, then they are equal on its substructure closure. -/
theorem eqOn_closure {f g : M →[L] N} {s : Set M} (h : Set.EqOn f g s) :
    Set.EqOn f g (closure L s) :=
  show closure L s ≤ f.eqLocus g from closure_le.2 h


theorem eq_of_eqOn_top {f g : M →[L] N} (h : Set.EqOn f g (⊤ : Substructure L M)) : f = g :=
  ext fun _ => h trivial


theorem eq_of_eqOn_dense (hs : closure L s = ⊤) {f g : M →[L] N} (h : s.EqOn f g) : f = g :=
  eq_of_eqOn_top <| hs ▸ eqOn_closure h


/-- The restriction of a first-order embedding to a substructure `s ⊆ M` gives an embedding `s → N`.
-/
def domRestrict (f : M ↪[L] N) (p : L.Substructure M) : p ↪[L] N :=
  f.comp p.subtype


@[simp]
theorem domRestrict_apply (f : M ↪[L] N) (p : L.Substructure M) (x : p) : f.domRestrict p x = f x :=
  rfl


/-- A first-order embedding `f : M → N` whose values lie in a substructure `p ⊆ N` can be restricted
to an embedding `M → p`. -/
def codRestrict (p : L.Substructure N) (f : M ↪[L] N) (h : ∀ c, f c ∈ p) : M ↪[L] p where
  toFun := f.toHom.codRestrict p h
  inj' _ _ ab := f.injective (Subtype.mk_eq_mk.1 ab)
  map_fun' {_} F x := (f.toHom.codRestrict p h).map_fun' F x
  map_rel' {n} r x := by
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S : L.Substructure M
      p : L.Substructure N
      f : L.Embedding M N
      h : ∀ (c : M), Membership.mem p (f c)
      n : Nat
      r : L.Relations n
      x : Fin n → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp { toFun := ⇑(Firs …
    -/
    simp only
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S : L.Substructure M
      p : L.Substructure N
      f : L.Embedding M N
      h : ∀ (c : M), Membership.mem p (f c)
      n : Nat
      r : L.Relations n
      x : Fin n → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp (⇑(FirstOrder.Lan …
    -/
    rw [← p.subtype.map_rel]
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S : L.Substructure M
      p : L.Substructure N
      f : L.Embedding M N
      h : ∀ (c : M), Membership.mem p (f c)
      n : Nat
      r : L.Relations n
      x : Fin n → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp (⇑p.subtype) (Fun …
    -/
    change RelMap r (Hom.comp p.subtype.toHom (f.toHom.codRestrict p h) ∘ x) ↔ _
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S : L.Substructure M
      p : L.Substructure N
      f : L.Embedding M N
      h : ∀ (c : M), Membership.mem p (f c)
      n : Nat
      r : L.Relations n
      x : Fin n → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp (⇑(p.subtype.toHo …
    -/
    rw [Hom.subtype_comp_codRestrict, ← f.map_rel]
    /-
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      P : Type u_2
      inst✝² : L.Structure M
      inst✝¹ : L.Structure N
      inst✝ : L.Structure P
      S : L.Substructure M
      p : L.Substructure N
      f : L.Embedding M N
      h : ∀ (c : M), Membership.mem p (f c)
      n : Nat
      r : L.Relations n
      x : Fin n → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap r (Function.comp (⇑f.toHom) x)) (F …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem codRestrict_apply (p : L.Substructure N) (f : M ↪[L] N) {h} (x : M) :
    (codRestrict p f h x : N) = f x :=
  rfl


@[simp]
theorem codRestrict_apply' (p : L.Substructure N) (f : M ↪[L] N) {h} (x : M) :
    codRestrict p f h x = ⟨f x, h x⟩ :=
  rfl


@[simp]
theorem comp_codRestrict (f : M ↪[L] N) (g : N ↪[L] P) (p : L.Substructure P) (h : ∀ b, g b ∈ p) :
    ((codRestrict p g h).comp f : M ↪[L] p) = codRestrict p (g.comp f) fun _ => h _ :=
  ext fun _ => rfl


@[simp]
theorem subtype_comp_codRestrict (f : M ↪[L] N) (p : L.Substructure N) (h : ∀ b, f b ∈ p) :
    p.subtype.comp (codRestrict p f h) = f :=
  ext fun _ => rfl


/-- The equivalence between a substructure `s` and its image `s.map f.toHom`, where `f` is an
  embedding. -/
noncomputable def substructureEquivMap (f : M ↪[L] N) (s : L.Substructure M) :
    s ≃[L] s.map f.toHom where
  toFun := codRestrict (s.map f.toHom) (f.domRestrict s) fun ⟨m, hm⟩ => ⟨m, hm, rfl⟩
  invFun n := ⟨Classical.choose n.2, (Classical.choose_spec n.2).1⟩
  left_inv := fun ⟨m, hm⟩ =>
    Subtype.mk_eq_mk.2
      (f.injective
        (Classical.choose_spec
            (codRestrict (s.map f.toHom) (f.domRestrict s) (fun ⟨m, hm⟩ => ⟨m, hm, rfl⟩)
                ⟨m, hm⟩).2).2)
  right_inv := fun ⟨_, hn⟩ => Subtype.mk_eq_mk.2 (Classical.choose_spec hn).2
                         /-
                           L : FirstOrder.Language
                           M : Type w
                           N : Type u_1
                           P : Type u_2
                           inst✝² : L.Structure M
                           inst✝¹ : L.Structure N
                           inst✝ : L.Structure P
                           S : L.Substructure M
                           f✝ : L.Embedding M N
                           s : L.Substructure M
                           n : Nat
                           f : L.Functions n
                           x : Fin n → Subtype fun x => Membership.mem s x
                           ⊢ Eq ({ toFun := ⇑(FirstOrder.Language.Embedding.codRestrict (FirstOrder.Langu …
                         -/
  map_fun' {n} f x := by aesop
                         /-
                           🎉 no goals
                         -/
                         /-
                           L : FirstOrder.Language
                           M : Type w
                           N : Type u_1
                           P : Type u_2
                           inst✝² : L.Structure M
                           inst✝¹ : L.Structure N
                           inst✝ : L.Structure P
                           S : L.Substructure M
                           f : L.Embedding M N
                           s : L.Substructure M
                           n : Nat
                           R : L.Relations n
                           x : Fin n → Subtype fun x => Membership.mem s x
                           ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp { toFun := ⇑(Firs …
                         -/
  map_rel' {n} R x := by aesop
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem substructureEquivMap_apply (f : M ↪[L] N) (p : L.Substructure M) (x : p) :
    (f.substructureEquivMap p x : N) = f x :=
  rfl


@[simp]
theorem subtype_substructureEquivMap (f : M ↪[L] N) (s : L.Substructure M) :
    (subtype _).comp (f.substructureEquivMap s).toEmbedding = f.comp (subtype _) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Embedding M N
    s : L.Substructure M
    ⊢ Eq ((FirstOrder.Language.Substructure.map f.toHom s).subtype.comp (f.substru …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- The equivalence between the domain and the range of an embedding `f`. -/
@[simps toEquiv_apply] noncomputable def equivRange (f : M ↪[L] N) : M ≃[L] f.toHom.range where
  toFun := codRestrict f.toHom.range f f.toHom.mem_range_self
  invFun n := Classical.choose n.2
  left_inv m :=
    f.injective (Classical.choose_spec (codRestrict f.toHom.range f f.toHom.mem_range_self m).2)
  right_inv := fun ⟨_, hn⟩ => Subtype.mk_eq_mk.2 (Classical.choose_spec hn)
                         /-
                           L : FirstOrder.Language
                           M : Type w
                           N : Type u_1
                           P : Type u_2
                           inst✝² : L.Structure M
                           inst✝¹ : L.Structure N
                           inst✝ : L.Structure P
                           S : L.Substructure M
                           f✝ : L.Embedding M N
                           n : Nat
                           f : L.Functions n
                           x : Fin n → M
                           ⊢ Eq ({ toFun := ⇑(FirstOrder.Language.Embedding.codRestrict f✝.toHom.range f✝ …
                         -/
  map_fun' {n} f x := by aesop
                         /-
                           🎉 no goals
                         -/
                         /-
                           L : FirstOrder.Language
                           M : Type w
                           N : Type u_1
                           P : Type u_2
                           inst✝² : L.Structure M
                           inst✝¹ : L.Structure N
                           inst✝ : L.Structure P
                           S : L.Substructure M
                           f : L.Embedding M N
                           n : Nat
                           R : L.Relations n
                           x : Fin n → M
                           ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp { toFun := ⇑(Firs …
                         -/
  map_rel' {n} R x := by aesop
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem equivRange_apply (f : M ↪[L] N) (x : M) : (f.equivRange x : N) = f x :=
  rfl


@[simp]
theorem subtype_equivRange (f : M ↪[L] N) : (subtype _).comp f.equivRange.toEmbedding = f := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Embedding M N
    ⊢ Eq (f.toHom.range.subtype.comp f.equivRange.toEmbedding) f
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


theorem toHom_range (f : M ≃[L] N) : f.toHom.range = ⊤ := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    ⊢ Eq f.toHom.range Top.top
  -/
  ext n
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    n : N
    ⊢ Iff (Membership.mem f.toHom.range n) (Membership.mem Top.top n)
  -/
  simp only [Hom.mem_range, coe_toHom, Substructure.mem_top, iff_true]
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    f : L.Equiv M N
    n : N
    ⊢ Exists fun y => Eq (f y) n
  -/
  exact ⟨f.symm n, apply_symm_apply _ _⟩
  /-
    🎉 no goals
  -/


/-- The embedding associated to an inclusion of substructures. -/
def inclusion {S T : L.Substructure M} (h : S ≤ T) : S ↪[L] T :=
  S.subtype.codRestrict _ fun x => h x.2


@[simp]
theorem inclusion_self (S : L.Substructure M) : inclusion (le_refl S) = Embedding.refl L S := rfl


@[simp]
theorem coe_inclusion {S T : L.Substructure M} (h : S ≤ T) :
    (inclusion h : S → T) = Set.inclusion h :=
  rfl


theorem range_subtype (S : L.Substructure M) : S.subtype.toHom.range = S := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    ⊢ Eq S.subtype.toHom.range S
  -/
  ext x
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    x : M
    ⊢ Iff (Membership.mem S.subtype.toHom.range x) (Membership.mem S x)
  -/
  simp only [Hom.mem_range, Embedding.coe_toHom, coeSubtype]
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    x : M
    ⊢ Iff (Exists fun y => Eq (↑y) x) (Membership.mem S x)
  -/
  refine ⟨?_, fun h => ⟨⟨x, h⟩, rfl⟩⟩
  /-
    case h
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    x : M
    ⊢ (Exists fun y => Eq (↑y) x) → Membership.mem S x
  -/
  rintro ⟨⟨y, hy⟩, rfl⟩
  /-
    case h.intro.mk
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    S : L.Substructure M
    y : M
    hy : Membership.mem S y
    ⊢ Membership.mem S ↑⟨y, hy⟩
  -/
  exact hy
  /-
    🎉 no goals
  -/


@[simp]
lemma subtype_comp_inclusion {S T : L.Substructure M} (h : S ≤ T) :
    T.subtype.comp (inclusion h) = S.subtype := rfl


