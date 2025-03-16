/-- Given a collection of setoids indexed by a type `ι`, a list `l` of indices, and a function that
  for each `i ∈ l` gives a term of the corresponding quotient type, then there is a corresponding
  term in the quotient of the product of the setoids indexed by `l`. -/
def listChoice {l : List ι} (q : ∀ i ∈ l, Quotient (S i)) : @Quotient (∀ i ∈ l, α i) piSetoid :=
  match l with
  |     [] => ⟦nofun⟧
  | i :: _ => Quotient.liftOn₂ (List.Pi.head (i := i) q)
    (listChoice (List.Pi.tail q))
    (⟦List.Pi.cons _ _ · ·⟧)
    (fun _ _ _ _ ha hl ↦ Quotient.sound (List.Pi.forall_rel_cons_ext ha hl))


theorem listChoice_mk {l : List ι} (a : ∀ i ∈ l, α i) : listChoice (S := S) (⟦a · ·⟧) = ⟦a⟧ :=
  match l with
  |     [] => Quotient.sound nofun
  | i :: l => by
    /-
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1
      ⊢ Eq (Quotient.listChoice fun x1 x2 => Quotient.mk (S x1) (a x1 x2)) (Quotient …
    -/
    unfold listChoice List.Pi.tail
    /-
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1
      ⊢ Eq ((List.Pi.head fun x1 x2 => Quotient.mk (S x1) (a x1 x2)).liftOn₂ (Quotie …
    -/
    rw [listChoice_mk]
    /-
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1
      ⊢ Eq ((List.Pi.head fun x1 x2 => Quotient.mk (S x1) (a x1 x2)).liftOn₂ (Quotie …
    -/
    exact congrArg (⟦·⟧) (List.Pi.cons_eta a)
    /-
      🎉 no goals
    -/


/-- Choice-free induction principle for quotients indexed by a `List`. -/
@[elab_as_elim]
lemma list_ind {l : List ι} {C : (∀ i ∈ l, Quotient (S i)) → Prop}
    (f : ∀ a : ∀ i ∈ l, α i, C (⟦a · ·⟧)) (q : ∀ i ∈ l, Quotient (S i)) : C q :=
  match l with
  |     [] => cast (congr_arg _ (funext₂ nofun)) (f nofun)
  | i :: l => by
    /-
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      C : ((i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)) → Prop
      f : ∀ (a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1), C fun x1 x …
      q : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)
      ⊢ C q
    -/
    rw [← List.Pi.cons_eta q]
    /-
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      C : ((i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)) → Prop
      f : ∀ (a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1), C fun x1 x …
      q : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)
      ⊢ C (List.Pi.cons i l (List.Pi.head q) (List.Pi.tail q))
    -/
    induction' List.Pi.head q using Quotient.ind with a
    /-
      case a
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      C : ((i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)) → Prop
      f : ∀ (a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1), C fun x1 x …
      q : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)
      a : α i
      ⊢ C (List.Pi.cons i l (Quotient.mk (S i) a) (List.Pi.tail q))
    -/
    refine @list_ind _ (fun q ↦ C (List.Pi.cons _ _ ⟦a⟧ q)) ?_ (List.Pi.tail q)
    /-
      case a
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      C : ((i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)) → Prop
      f : ∀ (a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1), C fun x1 x …
      q : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)
      a : α i
      ⊢ ∀ (a_1 : (i : ι) → Membership.mem l i → α i), (fun q => C (List.Pi.cons i l  …
    -/
    intro as
    /-
      case a
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      C : ((i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)) → Prop
      f : ∀ (a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1), C fun x1 x …
      q : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)
      a : α i
      as : (i : ι) → Membership.mem l i → α i
      ⊢ C (List.Pi.cons i l (Quotient.mk (S i) a) fun x1 x2 => Quotient.mk (S x1) (a …
    -/
    rw [List.Pi.cons_map a as (fun i ↦ Quotient.mk (S i))]
    /-
      case a
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      l✝ : List ι
      i : ι
      l : List ι
      C : ((i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)) → Prop
      f : ∀ (a : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → α i_1), C fun x1 x …
      q : (i_1 : ι) → Membership.mem (List.cons i l) i_1 → Quotient (S i_1)
      a : α i
      as : (i : ι) → Membership.mem l i → α i
      ⊢ C fun j hj => Quotient.mk (S j) (List.Pi.cons i l a as j hj)
    -/
    exact f _
    /-
      🎉 no goals
    -/


/-- Choice-free induction principle for quotients indexed by a finite type.
  See `Quotient.induction_on_pi` for the general version assuming `Classical.choice`. -/
@[elab_as_elim]
lemma ind_fintype_pi {C : (∀ i, Quotient (S i)) → Prop}
    (f : ∀ a : ∀ i, α i, C (⟦a ·⟧)) (q : ∀ i, Quotient (S i)) : C q := by
  have {m : Multiset ι} (C : (∀ i ∈ m, Quotient (S i)) → Prop) :
      ∀ (_ : ∀ a : ∀ i ∈ m, α i, C (⟦a · ·⟧)) (q : ∀ i ∈ m, Quotient (S i)), C q := by
    induction m using Quotient.ind
    exact list_ind
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Prop
    f : ∀ (a : (i : ι) → α i), C fun x => Quotient.mk (S x) (a x)
    q : (i : ι) → Quotient (S i)
    this : ∀ {m : Multiset ι} (C : ((i : ι) → Membership.mem m i → Quotient (S i)) …
    ⊢ C q
  -/
  exact this (fun q ↦ C (q · (Finset.mem_univ _))) (fun _ ↦ f _) (fun i _ ↦ q i)
  /-
    🎉 no goals
  -/


/-- Choice-free induction principle for quotients indexed by a finite type.
  See `Quotient.induction_on_pi` for the general version assuming `Classical.choice`. -/
@[elab_as_elim]
lemma induction_on_fintype_pi {C : (∀ i, Quotient (S i)) → Prop}
    (q : ∀ i, Quotient (S i)) (f : ∀ a : ∀ i, α i, C (⟦a ·⟧)) : C q :=
  ind_fintype_pi f q


/-- Given a collection of setoids indexed by a fintype `ι` and a function that for each `i : ι`
  gives a term of the corresponding quotient type, then there is corresponding term in the quotient
  of the product of the setoids.
  See `Quotient.choice` for the noncomputable general version. -/
def finChoice (q : ∀ i, Quotient (S i)) :
    @Quotient (∀ i, α i) piSetoid := by
  let e := Equiv.subtypeQuotientEquivQuotientSubtype (fun l : List ι ↦ ∀ i, i ∈ l)
    (fun s : Multiset ι ↦ ∀ i, i ∈ s) (fun i ↦ Iff.rfl) (fun _ _ ↦ Iff.rfl) ⟨_, Finset.mem_univ⟩
  refine e.liftOn
    (fun l ↦ (listChoice fun i _ ↦ q i).map (fun a i ↦ a i (l.2 i)) ?_) ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      β : Sort u_3
      q : (i : ι) → Quotient (S i)
      e : Quotient (Subtype.instSetoid_mathlib fun l => ∀ (i : ι), Membership.mem l  …
      l : Subtype fun l => ∀ (i : ι), Membership.mem l i
      ⊢ ∀ ⦃a b : (i : ι) → Membership.mem (↑l) i → α i⦄, HasEquiv.Equiv a b → HasEqu …
    -/
  · exact fun _ _ h i ↦ h i _
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    q : (i : ι) → Quotient (S i)
    e : Quotient (Subtype.instSetoid_mathlib fun l => ∀ (i : ι), Membership.mem l  …
    ⊢ ∀ (a b : Subtype fun l => ∀ (i : ι), Membership.mem l i), HasEquiv.Equiv a b …
  -/
  intro _ _ _
  /-
    case refine_2
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    q : (i : ι) → Quotient (S i)
    e : Quotient (Subtype.instSetoid_mathlib fun l => ∀ (i : ι), Membership.mem l  …
    a✝¹ b✝ : Subtype fun l => ∀ (i : ι), Membership.mem l i
    a✝ : HasEquiv.Equiv a✝¹ b✝
    ⊢ Eq ((fun l => Quotient.map (fun a i => a i ⋯) ⋯ (Quotient.listChoice fun i x …
  -/
  refine ind_fintype_pi (fun a ↦ ?_) q
  /-
    case refine_2
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    q : (i : ι) → Quotient (S i)
    e : Quotient (Subtype.instSetoid_mathlib fun l => ∀ (i : ι), Membership.mem l  …
    a✝¹ b✝ : Subtype fun l => ∀ (i : ι), Membership.mem l i
    a✝ : HasEquiv.Equiv a✝¹ b✝
    a : (i : ι) → α i
    ⊢ Eq ((fun l => Quotient.map (fun a i => a i ⋯) ⋯ (Quotient.listChoice fun i x …
  -/
  simp_rw [listChoice_mk, Quotient.map_mk]
  /-
    🎉 no goals
  -/


theorem finChoice_eq (a : ∀ i, α i) :
    finChoice (S := S) (⟦a ·⟧) = ⟦a⟧ := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    a : (i : ι) → α i
    ⊢ Eq (Quotient.finChoice fun x => Quotient.mk (S x) (a x)) (Quotient.mk piSeto …
  -/
  dsimp [finChoice]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    a : (i : ι) → α i
    ⊢ Eq (((Equiv.subtypeQuotientEquivQuotientSubtype (fun l => ∀ (i : ι), Members …
  -/
  obtain ⟨l, hl⟩ := (Finset.univ.val : Multiset ι).exists_rep
  /-
    case intro
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    a : (i : ι) → α i
    l : List ι
    hl : Eq (Quotient.mk (List.isSetoid ι) l) Finset.univ.val
    ⊢ Eq (((Equiv.subtypeQuotientEquivQuotientSubtype (fun l => ∀ (i : ι), Members …
  -/
  simp_rw [← hl, Equiv.subtypeQuotientEquivQuotientSubtype, listChoice_mk]
  /-
    case intro
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    a : (i : ι) → α i
    l : List ι
    hl : Eq (Quotient.mk (List.isSetoid ι) l) Finset.univ.val
    ⊢ Eq (({ toFun := fun a => Quotient.hrecOn (motive := fun x => (∀ (i : ι), Mem …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma eval_finChoice (f : ∀ i, Quotient (S i)) :
    eval (finChoice f) = f :=
                                        /-
                                          ι : Type u_1
                                          inst✝¹ : Fintype ι
                                          inst✝ : DecidableEq ι
                                          α : ι → Sort u_2
                                          S : (i : ι) → Setoid (α i)
                                          f : (i : ι) → Quotient (S i)
                                          a : (i : ι) → α i
                                          ⊢ Eq (Quotient.finChoice fun x => Quotient.mk (S x) (a x)).eval fun x => Quoti …
                                        -/
  induction_on_fintype_pi f (fun a ↦ by rw [finChoice_eq]; rfl)
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Lift a function on `∀ i, α i` to a function on `∀ i, Quotient (S i)`. -/
def finLiftOn (q : ∀ i, Quotient (S i)) (f : (∀ i, α i) → β)
    (h : ∀ (a b : ∀ i, α i), (∀ i, a i ≈ b i) → f a = f b) : β :=
  (finChoice q).liftOn f h


@[simp]
lemma finLiftOn_empty [e : IsEmpty ι] (q : ∀ i, Quotient (S i)) :
    finLiftOn (β := β) q = fun f _ ↦ f e.elim := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    e : IsEmpty ι
    q : (i : ι) → Quotient (S i)
    ⊢ Eq (Quotient.finLiftOn q) fun f x => f fun a => e.elim a
  -/
  ext f h
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    e : IsEmpty ι
    q : (i : ι) → Quotient (S i)
    f : ((i : ι) → α i) → β
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → Eq (f a …
    ⊢ Eq (Quotient.finLiftOn q f h) (f fun a => e.elim a)
  -/
  dsimp [finLiftOn]
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    e : IsEmpty ι
    q : (i : ι) → Quotient (S i)
    f : ((i : ι) → α i) → β
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → Eq (f a …
    ⊢ Eq ((Quotient.finChoice q).liftOn f h) (f fun a => e.elim a)
  -/
  induction finChoice q using Quotient.ind
  /-
    case h.h.a
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    e : IsEmpty ι
    q : (i : ι) → Quotient (S i)
    f : ((i : ι) → α i) → β
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → Eq (f a …
    a✝ : (i : ι) → α i
    ⊢ Eq ((Quotient.mk piSetoid a✝).liftOn f h) (f fun a => e.elim a)
  -/
  exact h _ _ e.elim
  /-
    🎉 no goals
  -/


@[simp]
lemma finLiftOn_mk (a : ∀ i, α i) :
    finLiftOn (S := S) (β := β) (⟦a ·⟧) = fun f _ ↦ f a := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    a : (i : ι) → α i
    ⊢ Eq (Quotient.finLiftOn fun x => Quotient.mk (S x) (a x)) fun f x => f a
  -/
  ext f h
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    a : (i : ι) → α i
    f : ((i : ι) → α i) → β
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → Eq (f a …
    ⊢ Eq (Quotient.finLiftOn (fun x => Quotient.mk (S x) (a x)) f h) (f a)
  -/
  dsimp [finLiftOn]
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    a : (i : ι) → α i
    f : ((i : ι) → α i) → β
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → Eq (f a …
    ⊢ Eq ((Quotient.finChoice fun x => Quotient.mk (S x) (a x)).liftOn f h) (f a)
  -/
  rw [finChoice_eq]
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    β : Sort u_3
    a : (i : ι) → α i
    f : ((i : ι) → α i) → β
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → Eq (f a …
    ⊢ Eq ((Quotient.mk piSetoid a).liftOn f h) (f a)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Quotient.finChoice` as an equivalence. -/
@[simps]
def finChoiceEquiv :
    (∀ i, Quotient (S i)) ≃ @Quotient (∀ i, α i) piSetoid where
  toFun := finChoice
  invFun := eval
  left_inv q := by
    /-
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      β : Sort u_3
      q : (i : ι) → Quotient (S i)
      ⊢ Eq (Quotient.finChoice q).eval q
    -/
    refine induction_on_fintype_pi q (fun a ↦ ?_)
    /-
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      β : Sort u_3
      q : (i : ι) → Quotient (S i)
      a : (i : ι) → α i
      ⊢ Eq (Quotient.finChoice fun x => Quotient.mk (S x) (a x)).eval fun x => Quoti …
    -/
    rw [finChoice_eq]
    /-
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      β : Sort u_3
      q : (i : ι) → Quotient (S i)
      a : (i : ι) → α i
      ⊢ Eq (Quotient.mk piSetoid a).eval fun x => Quotient.mk (S x) (a x)
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv q := by
    /-
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      β : Sort u_3
      q : Quotient piSetoid
      ⊢ Eq (Quotient.finChoice q.eval) q
    -/
    induction q using Quotient.ind
    /-
      case a
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      α : ι → Sort u_2
      S : (i : ι) → Setoid (α i)
      β : Sort u_3
      a✝ : (i : ι) → α i
      ⊢ Eq (Quotient.finChoice (Quotient.mk piSetoid a✝).eval) (Quotient.mk piSetoid …
    -/
    exact finChoice_eq _
    /-
      🎉 no goals
    -/


/-- Recursion principle for quotients indexed by a finite type. -/
@[elab_as_elim]
def finHRecOn {C : (∀ i, Quotient (S i)) → Sort*}
    (q : ∀ i, Quotient (S i))
    (f : ∀ a : ∀ i, α i, C (⟦a ·⟧))
    (h : ∀ (a b : ∀ i, α i), (∀ i, a i ≈ b i) → HEq (f a) (f b)) :
    C q :=
  eval_finChoice q ▸ (finChoice q).hrecOn f h


/-- Recursion principle for quotients indexed by a finite type. -/
@[elab_as_elim]
def finRecOn {C : (∀ i, Quotient (S i)) → Sort*}
    (q : ∀ i, Quotient (S i))
    (f : ∀ a : ∀ i, α i, C (⟦a ·⟧))
    (h : ∀ (a b : ∀ i, α i) (h : ∀ i, a i ≈ b i),
      Eq.ndrec (f a) (funext fun i ↦ Quotient.sound (h i)) = f b) :
    C q :=
  finHRecOn q f (rec_heq_iff_heq.mp <| heq_of_eq <| h · · ·)


@[simp]
lemma finHRecOn_mk {C : (∀ i, Quotient (S i)) → Sort*}
    (a : ∀ i, α i) :
    finHRecOn (C := C) (⟦a ·⟧) = fun f _ ↦ f a := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Sort u_4
    a : (i : ι) → α i
    ⊢ Eq (Quotient.finHRecOn fun x => Quotient.mk (S x) (a x)) fun f x => f a
  -/
  ext f h
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Sort u_4
    a : (i : ι) → α i
    f : (a : (i : ι) → α i) → C fun x => Quotient.mk (S x) (a x)
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → HEq (f  …
    ⊢ Eq (Quotient.finHRecOn (fun x => Quotient.mk (S x) (a x)) f h) (f a)
  -/
  refine eq_of_heq ((eqRec_heq _ _).trans ?_)
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Sort u_4
    a : (i : ι) → α i
    f : (a : (i : ι) → α i) → C fun x => Quotient.mk (S x) (a x)
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → HEq (f  …
    ⊢ HEq (Quotient.hrecOn (Quotient.finChoice fun x => Quotient.mk (S x) (a x)) f …
  -/
  rw [finChoice_eq]
  /-
    case h.h
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Sort u_4
    a : (i : ι) → α i
    f : (a : (i : ι) → α i) → C fun x => Quotient.mk (S x) (a x)
    h : ∀ (a b : (i : ι) → α i), (∀ (i : ι), HasEquiv.Equiv (a i) (b i)) → HEq (f  …
    ⊢ HEq (Quotient.hrecOn (Quotient.mk piSetoid a) f h) (f a)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma finRecOn_mk {C : (∀ i, Quotient (S i)) → Sort*}
    (a : ∀ i, α i) :
    finRecOn (C := C) (⟦a ·⟧) = fun f _ ↦ f a := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Sort u_4
    a : (i : ι) → α i
    ⊢ Eq (Quotient.finRecOn fun x => Quotient.mk (S x) (a x)) fun f x => f a
  -/
  unfold finRecOn
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    α : ι → Sort u_2
    S : (i : ι) → Setoid (α i)
    C : ((i : ι) → Quotient (S i)) → Sort u_4
    a : (i : ι) → α i
    ⊢ Eq (fun f h => Quotient.finHRecOn (fun x => Quotient.mk (S x) (a x)) f ⋯) fu …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a function that for each `i : ι` gives a term of the corresponding
truncation type, then there is corresponding term in the truncation of the product. -/
def finChoice (q : ∀ i, Trunc (α i)) : Trunc (∀ i, α i) :=
  Quotient.map' id (fun _ _ _ => trivial) (Quotient.finChoice q)


theorem finChoice_eq (f : ∀ i, α i) : (Trunc.finChoice fun i => Trunc.mk (f i)) = Trunc.mk f :=
  Subsingleton.elim _ _


/-- Lift a function on `∀ i, α i` to a function on `∀ i, Trunc (α i)`. -/
def finLiftOn (q : ∀ i, Trunc (α i)) (f : (∀ i, α i) → β) (h : ∀ (a b : ∀ i, α i), f a = f b) : β :=
  Quotient.finLiftOn q f (fun _ _ _ ↦ h _ _)


@[simp]
lemma finLiftOn_empty [e : IsEmpty ι] (q : ∀ i, Trunc (α i)) :
    finLiftOn (β := β) q = fun f _ ↦ f e.elim :=
  funext₂ fun _ _ ↦ congrFun₂ (Quotient.finLiftOn_empty q) _ _


@[simp]
lemma finLiftOn_mk (a : ∀ i, α i) :
    finLiftOn (β := β) (⟦a ·⟧) = fun f _ ↦ f a :=
  funext₂ fun _ _ ↦ congrFun₂ (Quotient.finLiftOn_mk a) _ _


/-- `Trunc.finChoice` as an equivalence. -/
@[simps]
def finChoiceEquiv : (∀ i, Trunc (α i)) ≃ Trunc (∀ i, α i) where
  toFun := finChoice
  invFun q i := q.map (· i)
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- Recursion principle for `Trunc`s indexed by a finite type. -/
@[elab_as_elim]
def finRecOn {C : (∀ i, Trunc (α i)) → Sort*}
    (q : ∀ i, Trunc (α i))
    (f : ∀ a : ∀ i, α i, C (mk <| a ·))
    (h : ∀ (a b : ∀ i, α i), (Eq.ndrec (f a) (funext fun _ ↦ Trunc.eq _ _)) = f b) :
    C q :=
  Quotient.finRecOn q (f ·) (fun _ _ _ ↦ h _ _)


@[simp]
lemma finRecOn_mk {C : (∀ i, Trunc (α i)) → Sort*}
    (a : ∀ i, α i) :
    finRecOn (C := C) (⟦a ·⟧) = fun f _ ↦ f a := by
  /-
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    α : ι → Sort u_2
    C : ((i : ι) → Trunc (α i)) → Sort u_4
    a : (i : ι) → α i
    ⊢ Eq (Trunc.finRecOn fun x => Quotient.mk trueSetoid (a x)) fun f x => f a
  -/
  unfold finRecOn
  /-
    ι : Type u_1
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    α : ι → Sort u_2
    C : ((i : ι) → Trunc (α i)) → Sort u_4
    a : (i : ι) → α i
    ⊢ Eq (fun f h => Quotient.finRecOn (fun x => Quotient.mk trueSetoid (a x)) (fu …
  -/
  simp
  /-
    🎉 no goals
  -/


