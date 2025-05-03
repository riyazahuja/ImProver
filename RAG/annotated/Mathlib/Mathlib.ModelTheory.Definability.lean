/-- A subset of a finite Cartesian product of a structure is definable over a set `A` when
  membership in the set is given by a first-order formula with parameters from `A`. -/
def Definable (s : Set (α → M)) : Prop :=
  ∃ φ : L[[A]].Formula α, s = setOf φ.Realize


theorem Definable.map_expansion {L' : FirstOrder.Language} [L'.Structure M] (h : A.Definable L s)
    (φ : L →ᴸ L') [φ.IsExpansionOn M] : A.Definable L' s := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝² : L.Structure M
    α : Type u₁
    s : Set (α → M)
    L' : FirstOrder.Language
    inst✝¹ : L'.Structure M
    h : A.Definable L s
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    ⊢ A.Definable L' s
  -/
  obtain ⟨ψ, rfl⟩ := h
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝² : L.Structure M
    α : Type u₁
    L' : FirstOrder.Language
    inst✝¹ : L'.Structure M
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    ψ : (L.withConstants ↑A).Formula α
    ⊢ A.Definable L' (setOf ψ.Realize)
  -/
  refine ⟨(φ.addConstants A).onFormula ψ, ?_⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝² : L.Structure M
    α : Type u₁
    L' : FirstOrder.Language
    inst✝¹ : L'.Structure M
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    ψ : (L.withConstants ↑A).Formula α
    ⊢ Eq (setOf ψ.Realize) (setOf ((FirstOrder.Language.LHom.addConstants (↑A) φ). …
  -/
  ext x
  /-
    case intro.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝² : L.Structure M
    α : Type u₁
    L' : FirstOrder.Language
    inst✝¹ : L'.Structure M
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    ψ : (L.withConstants ↑A).Formula α
    x : α → M
    ⊢ Iff (Membership.mem (setOf ψ.Realize) x) (Membership.mem (setOf ((FirstOrder …
  -/
  simp only [mem_setOf_eq, LHom.realize_onFormula]
  /-
    🎉 no goals
  -/


theorem definable_iff_exists_formula_sum :
    A.Definable L s ↔ ∃ φ : L.Formula (A ⊕ α), s = {v | φ.Realize (Sum.elim (↑) v)} := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    ⊢ Iff (A.Definable L s) (Exists fun φ => Eq s (setOf fun v => φ.Realize (Sum.e …
  -/
  rw [Definable, Equiv.exists_congr_left (BoundedFormula.constantsVarsEquiv)]
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    ⊢ Iff (Exists fun b => Eq s (setOf (FirstOrder.Language.Formula.Realize (First …
  -/
  refine exists_congr (fun φ => iff_iff_eq.2 (congr_arg (s = ·) ?_))
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : L.BoundedFormula (Sum (↑A) α) 0
    ⊢ Eq (setOf (FirstOrder.Language.Formula.Realize (FirstOrder.Language.BoundedF …
  -/
  ext
  simp only [BoundedFormula.constantsVarsEquiv, constantsOn,
    BoundedFormula.mapTermRelEquiv_symm_apply, mem_setOf_eq, Formula.Realize]
  /-
    case h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : L.BoundedFormula (Sum (↑A) α) 0
    x✝ : α → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.mapTermRel (fun n => ⇑FirstOrder.La …
  -/
  refine BoundedFormula.realize_mapTermRel_id ?_ (fun _ _ _ => rfl)
  /-
    case h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : L.BoundedFormula (Sum (↑A) α) 0
    x✝ : α → M
    ⊢ ∀ (n : Nat) (t : L.Term (Sum (Sum (↑A) α) (Fin n))) (xs : Fin n → M), Eq (Fi …
  -/
  intros
  simp only [Term.constantsVarsEquivLeft_symm_apply, Term.realize_varsToConstants,
    coe_con, Term.realize_relabel]
  /-
    case h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : L.BoundedFormula (Sum (↑A) α) 0
    x✝ : α → M
    n✝ : Nat
    t✝ : L.Term (Sum (Sum (↑A) α) (Fin n✝))
    xs✝ : Fin n✝ → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (Sum.elim (fun a => ↑a)  …
  -/
  congr
  /-
    case h.e_v
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : L.BoundedFormula (Sum (↑A) α) 0
    x✝ : α → M
    n✝ : Nat
    t✝ : L.Term (Sum (Sum (↑A) α) (Fin n✝))
    xs✝ : Fin n✝ → M
    ⊢ Eq (Function.comp (Sum.elim (fun a => ↑a) (Sum.elim x✝ xs✝)) ⇑(Equiv.sumAsso …
  -/
  ext a
  /-
    case h.e_v.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : L.BoundedFormula (Sum (↑A) α) 0
    x✝ : α → M
    n✝ : Nat
    t✝ : L.Term (Sum (Sum (↑A) α) (Fin n✝))
    xs✝ : Fin n✝ → M
    a : Sum (Sum (↑A) α) (Fin n✝)
    ⊢ Eq (Function.comp (Sum.elim (fun a => ↑a) (Sum.elim x✝ xs✝)) (⇑(Equiv.sumAss …
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases a with (_ | _) | _ <;> rfl
                                /-
                                  🎉 no goals
                                -/


theorem empty_definable_iff :
    (∅ : Set M).Definable L s ↔ ∃ φ : L.Formula α, s = setOf φ.Realize := by
  /-
    M : Type w
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    ⊢ Iff (EmptyCollection.emptyCollection.Definable L s) (Exists fun φ => Eq s (s …
  -/
  rw [Definable, Equiv.exists_congr_left (LEquiv.addEmptyConstants L (∅ : Set M)).onFormula]
  /-
    M : Type w
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    ⊢ Iff (Exists fun φ => Eq s (setOf φ.Realize)) (Exists fun b => Eq s (setOf (( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem definable_iff_empty_definable_with_params :
    A.Definable L s ↔ (∅ : Set M).Definable (L[[A]]) s :=
  empty_definable_iff.symm


theorem Definable.mono (hAs : A.Definable L s) (hAB : A ⊆ B) : B.Definable L s := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    B : Set M
    s : Set (α → M)
    hAs : A.Definable L s
    hAB : HasSubset.Subset A B
    ⊢ B.Definable L s
  -/
  rw [definable_iff_empty_definable_with_params] at *
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    B : Set M
    s : Set (α → M)
    hAs : EmptyCollection.emptyCollection.Definable (L.withConstants ↑A) s
    hAB : HasSubset.Subset A B
    ⊢ EmptyCollection.emptyCollection.Definable (L.withConstants ↑B) s
  -/
  exact hAs.map_expansion (L.lhomWithConstantsMap (Set.inclusion hAB))
  /-
    🎉 no goals
  -/


@[simp]
theorem definable_empty : A.Definable L (∅ : Set (α → M)) :=
  ⟨⊥, by
    /-
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      ⊢ Eq EmptyCollection.emptyCollection (setOf Bot.bot.Realize)
    -/
    ext
    /-
      case h
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      x✝ : α → M
      ⊢ Iff (Membership.mem EmptyCollection.emptyCollection x✝) (Membership.mem (set …
    -/
    simp⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem definable_univ : A.Definable L (univ : Set (α → M)) :=
  ⟨⊤, by
    /-
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      ⊢ Eq Set.univ (setOf Top.top.Realize)
    -/
    ext
    /-
      case h
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      x✝ : α → M
      ⊢ Iff (Membership.mem Set.univ x✝) (Membership.mem (setOf Top.top.Realize) x✝)
    -/
    simp⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem Definable.inter {f g : Set (α → M)} (hf : A.Definable L f) (hg : A.Definable L g) :
    A.Definable L (f ∩ g) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    f g : Set (α → M)
    hf : A.Definable L f
    hg : A.Definable L g
    ⊢ A.Definable L (Inter.inter f g)
  -/
  rcases hf with ⟨φ, rfl⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    g : Set (α → M)
    hg : A.Definable L g
    φ : (L.withConstants ↑A).Formula α
    ⊢ A.Definable L (Inter.inter (setOf φ.Realize) g)
  -/
  rcases hg with ⟨θ, rfl⟩
  /-
    case intro.intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    φ θ : (L.withConstants ↑A).Formula α
    ⊢ A.Definable L (Inter.inter (setOf φ.Realize) (setOf θ.Realize))
  -/
  refine ⟨φ ⊓ θ, ?_⟩
  /-
    case intro.intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    φ θ : (L.withConstants ↑A).Formula α
    ⊢ Eq (Inter.inter (setOf φ.Realize) (setOf θ.Realize)) (setOf (Min.min φ θ).Re …
  -/
  ext
  /-
    case intro.intro.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    φ θ : (L.withConstants ↑A).Formula α
    x✝ : α → M
    ⊢ Iff (Membership.mem (Inter.inter (setOf φ.Realize) (setOf θ.Realize)) x✝) (M …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem Definable.union {f g : Set (α → M)} (hf : A.Definable L f) (hg : A.Definable L g) :
    A.Definable L (f ∪ g) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    f g : Set (α → M)
    hf : A.Definable L f
    hg : A.Definable L g
    ⊢ A.Definable L (Union.union f g)
  -/
  rcases hf with ⟨φ, hφ⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    f g : Set (α → M)
    hg : A.Definable L g
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq f (setOf φ.Realize)
    ⊢ A.Definable L (Union.union f g)
  -/
  rcases hg with ⟨θ, hθ⟩
  /-
    case intro.intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    f g : Set (α → M)
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq f (setOf φ.Realize)
    θ : (L.withConstants ↑A).Formula α
    hθ : Eq g (setOf θ.Realize)
    ⊢ A.Definable L (Union.union f g)
  -/
  refine ⟨φ ⊔ θ, ?_⟩
  /-
    case intro.intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    f g : Set (α → M)
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq f (setOf φ.Realize)
    θ : (L.withConstants ↑A).Formula α
    hθ : Eq g (setOf θ.Realize)
    ⊢ Eq (Union.union f g) (setOf (Max.max φ θ).Realize)
  -/
  ext
  /-
    case intro.intro.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    f g : Set (α → M)
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq f (setOf φ.Realize)
    θ : (L.withConstants ↑A).Formula α
    hθ : Eq g (setOf θ.Realize)
    x✝ : α → M
    ⊢ Iff (Membership.mem (Union.union f g) x✝) (Membership.mem (setOf (Max.max φ  …
  -/
  rw [hφ, hθ, mem_setOf_eq, Formula.realize_sup, mem_union, mem_setOf_eq, mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem definable_finset_inf {ι : Type*} {f : ι → Set (α → M)} (hf : ∀ i, A.Definable L (f i))
    (s : Finset ι) : A.Definable L (s.inf f) := by
  classical
    refine Finset.induction definable_univ (fun i s _ h => ?_) s
    rw [Finset.inf_insert]
    exact (hf i).inter h


theorem definable_finset_sup {ι : Type*} {f : ι → Set (α → M)} (hf : ∀ i, A.Definable L (f i))
    (s : Finset ι) : A.Definable L (s.sup f) := by
  classical
    refine Finset.induction definable_empty (fun i s _ h => ?_) s
    rw [Finset.sup_insert]
    exact (hf i).union h


theorem definable_finset_biInter {ι : Type*} {f : ι → Set (α → M)}
    (hf : ∀ i, A.Definable L (f i)) (s : Finset ι) : A.Definable L (⋂ i ∈ s, f i) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    ι : Type u_2
    f : ι → Set (α → M)
    hf : ∀ (i : ι), A.Definable L (f i)
    s : Finset ι
    ⊢ A.Definable L (Set.iInter fun i => Set.iInter fun h => f i)
  -/
  rw [← Finset.inf_set_eq_iInter]
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    ι : Type u_2
    f : ι → Set (α → M)
    hf : ∀ (i : ι), A.Definable L (f i)
    s : Finset ι
    ⊢ A.Definable L (s.inf f)
  -/
  exact definable_finset_inf hf s
  /-
    🎉 no goals
  -/


theorem definable_finset_biUnion {ι : Type*} {f : ι → Set (α → M)}
    (hf : ∀ i, A.Definable L (f i)) (s : Finset ι) : A.Definable L (⋃ i ∈ s, f i) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    ι : Type u_2
    f : ι → Set (α → M)
    hf : ∀ (i : ι), A.Definable L (f i)
    s : Finset ι
    ⊢ A.Definable L (Set.iUnion fun i => Set.iUnion fun h => f i)
  -/
  rw [← Finset.sup_set_eq_biUnion]
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    ι : Type u_2
    f : ι → Set (α → M)
    hf : ∀ (i : ι), A.Definable L (f i)
    s : Finset ι
    ⊢ A.Definable L (s.sup f)
  -/
  exact definable_finset_sup hf s
  /-
    🎉 no goals
  -/


@[simp]
theorem Definable.compl {s : Set (α → M)} (hf : A.Definable L s) : A.Definable L sᶜ := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    hf : A.Definable L s
    ⊢ A.Definable L (HasCompl.compl s)
  -/
  rcases hf with ⟨φ, hφ⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq s (setOf φ.Realize)
    ⊢ A.Definable L (HasCompl.compl s)
  -/
  refine ⟨φ.not, ?_⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq s (setOf φ.Realize)
    ⊢ Eq (HasCompl.compl s) (setOf φ.not.Realize)
  -/
  ext v
  /-
    case intro.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    s : Set (α → M)
    φ : (L.withConstants ↑A).Formula α
    hφ : Eq s (setOf φ.Realize)
    v : α → M
    ⊢ Iff (Membership.mem (HasCompl.compl s) v) (Membership.mem (setOf φ.not.Reali …
  -/
  rw [hφ, compl_setOf, mem_setOf, mem_setOf, Formula.realize_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem Definable.sdiff {s t : Set (α → M)} (hs : A.Definable L s) (ht : A.Definable L t) :
    A.Definable L (s \ t) :=
  hs.inter ht.compl


@[simp] lemma Definable.himp {s t : Set (α → M)} (hs : A.Definable L s) (ht : A.Definable L t) :
                                /-
                                  M : Type w
                                  A : Set M
                                  L : FirstOrder.Language
                                  inst✝ : L.Structure M
                                  α : Type u₁
                                  s t : Set (α → M)
                                  hs : A.Definable L s
                                  ht : A.Definable L t
                                  ⊢ A.Definable L (HImp.himp s t)
                                -/
    A.Definable L (s ⇨ t) := by rw [himp_eq]; exact ht.union hs.compl
                                              /-
                                                🎉 no goals
                                              -/


theorem Definable.preimage_comp (f : α → β) {s : Set (α → M)} (h : A.Definable L s) :
    A.Definable L ((fun g : β → M => g ∘ f) ⁻¹' s) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    β : Type u_1
    f : α → β
    s : Set (α → M)
    h : A.Definable L s
    ⊢ A.Definable L (Set.preimage (fun g => Function.comp g f) s)
  -/
  obtain ⟨φ, rfl⟩ := h
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    β : Type u_1
    f : α → β
    φ : (L.withConstants ↑A).Formula α
    ⊢ A.Definable L (Set.preimage (fun g => Function.comp g f) (setOf φ.Realize))
  -/
  refine ⟨φ.relabel f, ?_⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    β : Type u_1
    f : α → β
    φ : (L.withConstants ↑A).Formula α
    ⊢ Eq (Set.preimage (fun g => Function.comp g f) (setOf φ.Realize)) (setOf (Fir …
  -/
  ext
  /-
    case intro.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    β : Type u_1
    f : α → β
    φ : (L.withConstants ↑A).Formula α
    x✝ : β → M
    ⊢ Iff (Membership.mem (Set.preimage (fun g => Function.comp g f) (setOf φ.Real …
  -/
  simp only [Set.preimage_setOf_eq, mem_setOf_eq, Formula.realize_relabel]
  /-
    🎉 no goals
  -/


theorem Definable.image_comp_equiv {s : Set (β → M)} (h : A.Definable L s) (f : α ≃ β) :
    A.Definable L ((fun g : β → M => g ∘ f) '' s) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    β : Type u_1
    s : Set (β → M)
    h : A.Definable L s
    f : Equiv α β
    ⊢ A.Definable L (Set.image (fun g => Function.comp g ⇑f) s)
  -/
  refine (congr rfl ?_).mp (h.preimage_comp f.symm)
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    β : Type u_1
    s : Set (β → M)
    h : A.Definable L s
    f : Equiv α β
    ⊢ Eq (Set.preimage (fun g => Function.comp g ⇑f.symm) s) (Set.image (fun g =>  …
  -/
  rw [image_eq_preimage_of_inverse]
    /-
      case h₁
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      β : Type u_1
      s : Set (β → M)
      h : A.Definable L s
      f : Equiv α β
      ⊢ Function.LeftInverse (fun g => Function.comp g ⇑f.symm) fun g => Function.co …
    -/
  · intro i
    /-
      case h₁
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      β : Type u_1
      s : Set (β → M)
      h : A.Definable L s
      f : Equiv α β
      i : β → M
      ⊢ Eq ((fun g => Function.comp g ⇑f.symm) ((fun g => Function.comp g ⇑f) i)) i
    -/
    ext b
    /-
      case h₁.h
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      β : Type u_1
      s : Set (β → M)
      h : A.Definable L s
      f : Equiv α β
      i : β → M
      b : β
      ⊢ Eq ((fun g => Function.comp g ⇑f.symm) ((fun g => Function.comp g ⇑f) i) b)  …
    -/
    simp only [Function.comp_apply, Equiv.apply_symm_apply]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      β : Type u_1
      s : Set (β → M)
      h : A.Definable L s
      f : Equiv α β
      ⊢ Function.RightInverse (fun g => Function.comp g ⇑f.symm) fun g => Function.c …
    -/
  · intro i
    /-
      case h₂
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      β : Type u_1
      s : Set (β → M)
      h : A.Definable L s
      f : Equiv α β
      i : α → M
      ⊢ Eq ((fun g => Function.comp g ⇑f) ((fun g => Function.comp g ⇑f.symm) i)) i
    -/
    ext a
    /-
      case h₂.h
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      β : Type u_1
      s : Set (β → M)
      h : A.Definable L s
      f : Equiv α β
      i : α → M
      a : α
      ⊢ Eq ((fun g => Function.comp g ⇑f) ((fun g => Function.comp g ⇑f.symm) i) a)  …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem definable_iff_finitely_definable :
    A.Definable L s ↔ ∃ (A0 : Finset M), (A0 : Set M) ⊆ A ∧
      (A0 : Set M).Definable L s := by
  classical
  constructor
  · simp only [definable_iff_exists_formula_sum]
    rintro ⟨φ, rfl⟩
    let A0 := (φ.freeVarFinset.toLeft).image Subtype.val
    refine ⟨A0, by simp [A0], (φ.restrictFreeVar <| fun x => Sum.casesOn x.1
        (fun x hx => Sum.inl ⟨x, by simp [A0, hx]⟩) (fun x _ => Sum.inr x) x.2), ?_⟩
    ext
    simp only [Formula.Realize, mem_setOf_eq, Finset.coe_sort_coe]
    exact iff_comm.1 <| BoundedFormula.realize_restrictFreeVar _ (by simp)
  · rintro ⟨A0, hA0, hd⟩
    exact Definable.mono hd hA0


/-- This lemma is only intended as a helper for `Definable.image_comp`. -/
theorem Definable.image_comp_sum_inl_fin (m : ℕ) {s : Set (Sum α (Fin m) → M)}
    (h : A.Definable L s) : A.Definable L ((fun g : Sum α (Fin m) → M => g ∘ Sum.inl) '' s) := by
  /-
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    m : Nat
    s : Set (Sum α (Fin m) → M)
    h : A.Definable L s
    ⊢ A.Definable L (Set.image (fun g => Function.comp g Sum.inl) s)
  -/
  obtain ⟨φ, rfl⟩ := h
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    m : Nat
    φ : (L.withConstants ↑A).Formula (Sum α (Fin m))
    ⊢ A.Definable L (Set.image (fun g => Function.comp g Sum.inl) (setOf φ.Realize))
  -/
  refine ⟨(BoundedFormula.relabel id φ).exs, ?_⟩
  /-
    case intro
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    m : Nat
    φ : (L.withConstants ↑A).Formula (Sum α (Fin m))
    ⊢ Eq (Set.image (fun g => Function.comp g Sum.inl) (setOf φ.Realize)) (setOf ( …
  -/
  ext x
  simp only [Set.mem_image, mem_setOf_eq, BoundedFormula.realize_exs,
    BoundedFormula.realize_relabel, Function.comp_id, Fin.castAdd_zero, Fin.cast_refl]
  /-
    case intro.h
    M : Type w
    A : Set M
    L : FirstOrder.Language
    inst✝ : L.Structure M
    α : Type u₁
    m : Nat
    φ : (L.withConstants ↑A).Formula (Sum α (Fin m))
    x : α → M
    ⊢ Iff (Exists fun x_1 => And (φ.Realize x_1) (Eq (Function.comp x_1 Sum.inl) x …
  -/
  constructor
    /-
      case intro.h.mp
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      m : Nat
      φ : (L.withConstants ↑A).Formula (Sum α (Fin m))
      x : α → M
      ⊢ (Exists fun x_1 => And (φ.Realize x_1) (Eq (Function.comp x_1 Sum.inl) x)) → …
    -/
  · rintro ⟨y, hy, rfl⟩
    exact
      ⟨y ∘ Sum.inr, (congr (congr rfl (Sum.elim_comp_inl_inr y).symm) (funext finZeroElim)).mp hy⟩
    /-
      case intro.h.mpr
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      m : Nat
      φ : (L.withConstants ↑A).Formula (Sum α (Fin m))
      x : α → M
      ⊢ (Exists fun xs => FirstOrder.Language.BoundedFormula.Realize φ (Sum.elim x x …
    -/
  · rintro ⟨y, hy⟩
    /-
      case intro.h.mpr.intro
      M : Type w
      A : Set M
      L : FirstOrder.Language
      inst✝ : L.Structure M
      α : Type u₁
      m : Nat
      φ : (L.withConstants ↑A).Formula (Sum α (Fin m))
      x : α → M
      y : Fin (HAdd.hAdd m 0) → M
      hy : FirstOrder.Language.BoundedFormula.Realize φ (Sum.elim x y) (Function.com …
      ⊢ Exists fun x_1 => And (φ.Realize x_1) (Eq (Function.comp x_1 Sum.inl) x)
    -/
    exact ⟨Sum.elim x y, (congr rfl (funext finZeroElim)).mp hy, Sum.elim_comp_inl _ _⟩
    /-
      🎉 no goals
    -/


/-- Shows that definability is closed under finite projections. -/
theorem Definable.image_comp_embedding {s : Set (β → M)} (h : A.Definable L s) (f : α ↪ β)
    [Finite β] : A.Definable L ((fun g : β → M => g ∘ f) '' s) := by
  classical
    cases nonempty_fintype β
    refine
      (congr rfl (ext fun x => ?_)).mp
        (((h.image_comp_equiv (Equiv.Set.sumCompl (range f))).image_comp_equiv
              (Equiv.sumCongr (Equiv.ofInjective f f.injective)
                (Fintype.equivFin (↥(range f)ᶜ)).symm)).image_comp_sum_inl_fin
          _)
    simp only [mem_preimage, mem_image, exists_exists_and_eq_and]
    refine exists_congr fun y => and_congr_right fun _ => Eq.congr_left (funext fun a => ?_)
    simp


/-- Shows that definability is closed under finite projections. -/
theorem Definable.image_comp {s : Set (β → M)} (h : A.Definable L s) (f : α → β) [Finite α]
    [Finite β] : A.Definable L ((fun g : β → M => g ∘ f) '' s) := by
  classical
    cases nonempty_fintype α
    cases nonempty_fintype β
    have h :=
      (((h.image_comp_equiv (Equiv.Set.sumCompl (range f))).image_comp_equiv
                (Equiv.sumCongr (_root_.Equiv.refl _)
                  (Fintype.equivFin _).symm)).image_comp_sum_inl_fin
            _).preimage_comp
        (rangeSplitting f)
    have h' :
      A.Definable L { x : α → M | ∀ a, x a = x (rangeSplitting f (rangeFactorization f a)) } := by
      have h' : ∀ a,
        A.Definable L { x : α → M | x a = x (rangeSplitting f (rangeFactorization f a)) } := by
          refine fun a => ⟨(var a).equal (var (rangeSplitting f (rangeFactorization f a))), ext ?_⟩
          simp
      refine (congr rfl (ext ?_)).mp (definable_finset_biInter h' Finset.univ)
      simp
    refine (congr rfl (ext fun x => ?_)).mp (h.inter h')
    simp only [Equiv.coe_trans, mem_inter_iff, mem_preimage, mem_image, exists_exists_and_eq_and,
      mem_setOf_eq]
    constructor
    · rintro ⟨⟨y, ys, hy⟩, hx⟩
      refine ⟨y, ys, ?_⟩
      ext a
      rw [hx a, ← Function.comp_apply (f := x), ← hy]
      simp
    · rintro ⟨y, ys, rfl⟩
      refine ⟨⟨y, ys, ?_⟩, fun a => ?_⟩
      · ext
        simp [Set.apply_rangeSplitting f]
      · rw [Function.comp_apply, Function.comp_apply, apply_rangeSplitting f,
          rangeFactorization_coe]


/-- A 1-dimensional version of `Definable`, for `Set M`. -/
def Definable₁ (s : Set M) : Prop :=
  A.Definable L { x : Fin 1 → M | x 0 ∈ s }


/-- A 2-dimensional version of `Definable`, for `Set (M × M)`. -/
def Definable₂ (s : Set (M × M)) : Prop :=
  A.Definable L { x : Fin 2 → M | (x 0, x 1) ∈ s }


/-- Definable sets are subsets of finite Cartesian products of a structure such that membership is
  given by a first-order formula. -/
def DefinableSet :=
  { s : Set (α → M) // A.Definable L s }


instance instSetLike : SetLike (L.DefinableSet A α) (α → M) where
  coe := Subtype.val
  coe_injective' := Subtype.val_injective


instance instTop : Top (L.DefinableSet A α) :=
  ⟨⟨⊤, definable_univ⟩⟩


instance instBot : Bot (L.DefinableSet A α) :=
  ⟨⟨⊥, definable_empty⟩⟩


instance instSup : Max (L.DefinableSet A α) :=
  ⟨fun s t => ⟨s ∪ t, s.2.union t.2⟩⟩


instance instInf : Min (L.DefinableSet A α) :=
  ⟨fun s t => ⟨s ∩ t, s.2.inter t.2⟩⟩


instance instHasCompl : HasCompl (L.DefinableSet A α) :=
  ⟨fun s => ⟨sᶜ, s.2.compl⟩⟩


instance instSDiff : SDiff (L.DefinableSet A α) :=
  ⟨fun s t => ⟨s \ t, s.2.sdiff t.2⟩⟩

-- Why does it complain that `s ⇨ t` is noncomputable?

noncomputable instance instHImp : HImp (L.DefinableSet A α) where
  himp s t := ⟨s ⇨ t, s.2.himp t.2⟩


instance instInhabited : Inhabited (L.DefinableSet A α) :=
  ⟨⊥⟩


theorem le_iff : s ≤ t ↔ (s : Set (α → M)) ≤ (t : Set (α → M)) :=
  Iff.rfl


@[simp]
theorem mem_top : x ∈ (⊤ : L.DefinableSet A α) :=
  mem_univ x


@[simp]
theorem not_mem_bot {x : α → M} : ¬x ∈ (⊥ : L.DefinableSet A α) :=
  not_mem_empty x


@[simp]
theorem mem_sup : x ∈ s ⊔ t ↔ x ∈ s ∨ x ∈ t :=
  Iff.rfl


@[simp]
theorem mem_inf : x ∈ s ⊓ t ↔ x ∈ s ∧ x ∈ t :=
  Iff.rfl


@[simp]
theorem mem_compl : x ∈ sᶜ ↔ ¬x ∈ s :=
  Iff.rfl


@[simp]
theorem mem_sdiff : x ∈ s \ t ↔ x ∈ s ∧ ¬x ∈ t :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_top : ((⊤ : L.DefinableSet A α) : Set (α → M)) = univ :=
  rfl


@[simp, norm_cast]
theorem coe_bot : ((⊥ : L.DefinableSet A α) : Set (α → M)) = ∅ :=
  rfl


@[simp, norm_cast]
theorem coe_sup (s t : L.DefinableSet A α) :
    ((s ⊔ t : L.DefinableSet A α) : Set (α → M)) = (s : Set (α → M)) ∪ (t : Set (α → M)) :=
  rfl


@[simp, norm_cast]
theorem coe_inf (s t : L.DefinableSet A α) :
    ((s ⊓ t : L.DefinableSet A α) : Set (α → M)) = (s : Set (α → M)) ∩ (t : Set (α → M)) :=
  rfl


@[simp, norm_cast]
theorem coe_compl (s : L.DefinableSet A α) :
    ((sᶜ : L.DefinableSet A α) : Set (α → M)) = (s : Set (α → M))ᶜ :=
  rfl


@[simp, norm_cast]
theorem coe_sdiff (s t : L.DefinableSet A α) :
    ((s \ t : L.DefinableSet A α) : Set (α → M)) = (s : Set (α → M)) \ (t : Set (α → M)) :=
  rfl


@[simp, norm_cast]
lemma coe_himp (s t : L.DefinableSet A α) : ↑(s ⇨ t) = (s ⇨ t : Set (α → M)) := rfl


noncomputable instance instBooleanAlgebra : BooleanAlgebra (L.DefinableSet A α) :=
  Function.Injective.booleanAlgebra (α := L.DefinableSet A α) _ Subtype.coe_injective
    coe_sup coe_inf coe_top coe_bot coe_compl coe_sdiff coe_himp


