/-- If `s` is a linear independent set of vectors, we can extend it to a basis. -/
noncomputable def extend (hs : LinearIndependent K ((↑) : s → V)) :
    Basis (hs.extend (subset_univ s)) K V :=
  Basis.mk
    (@LinearIndependent.restrict_of_comp_subtype _ _ _ id _ _ _ _ (hs.linearIndependent_extend _))
                                     /-
                                       ι : Type u_1
                                       ι' : Type u_2
                                       K : Type u_3
                                       V : Type u_4
                                       V' : Type u_5
                                       inst✝⁴ : DivisionRing K
                                       inst✝³ : AddCommGroup V
                                       inst✝² : AddCommGroup V'
                                       inst✝¹ : Module K V
                                       inst✝ : Module K V'
                                       v : ι → V
                                       s t : Set V
                                       x y z : V
                                       hs : LinearIndependent K Subtype.val
                                       ⊢ HasSubset.Subset ↑Top.top ↑(Submodule.span K (Set.range ((hs.extend ⋯).restr …
                                     -/
    (SetLike.coe_subset_coe.mp <| by simpa using hs.subset_span_extend (subset_univ s))
                                     /-
                                       🎉 no goals
                                     -/


theorem extend_apply_self (hs : LinearIndependent K ((↑) : s → V)) (x : hs.extend _) :
    Basis.extend hs x = x :=
  Basis.mk_apply _ _ _


@[simp]
theorem coe_extend (hs : LinearIndependent K ((↑) : s → V)) : ⇑(Basis.extend hs) = ((↑) : _ → _) :=
  funext (extend_apply_self hs)


theorem range_extend (hs : LinearIndependent K ((↑) : s → V)) :
    range (Basis.extend hs) = hs.extend (subset_univ _) := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    hs : LinearIndependent K Subtype.val
    ⊢ Eq (Set.range ⇑(Basis.extend hs)) (hs.extend ⋯)
  -/
  rw [coe_extend, Subtype.range_coe_subtype, setOf_mem_eq]
  /-
    🎉 no goals
  -/

-- Porting note: adding this to make the statement of `subExtend` more readable

/-- Auxiliary definition: the index for the new basis vectors in `Basis.sumExtend`.

The specific value of this definition should be considered an implementation detail.
-/
def sumExtendIndex (hs : LinearIndependent K v) : Set V :=
  LinearIndependent.extend hs.to_subtype_range (subset_univ _) \ range v


/-- If `v` is a linear independent family of vectors, extend it to a basis indexed by a sum type. -/
noncomputable def sumExtend (hs : LinearIndependent K v) : Basis (ι ⊕ sumExtendIndex hs) K V :=
  let s := Set.range v
  let e : ι ≃ s := Equiv.ofInjective v hs.injective
  let b := hs.to_subtype_range.extend (subset_univ (Set.range v))
  (Basis.extend hs.to_subtype_range).reindex <|
    Equiv.symm <|
      calc
        ι ⊕ (b \ s : Set V) ≃ s ⊕ (b \ s : Set V) := Equiv.sumCongr e (Equiv.refl _)
        _ ≃ b :=
          haveI := Classical.decPred (· ∈ s)
          Equiv.Set.sumDiffSubset (hs.to_subtype_range.subset_extend _)


theorem subset_extend {s : Set V} (hs : LinearIndependent K ((↑) : s → V)) :
    s ⊆ hs.extend (Set.subset_univ _) :=
  hs.subset_extend _


/-- If `s` is a family of linearly independent vectors contained in a set `t` spanning `V`,
then one can get a basis of `V` containing `s` and contained in `t`. -/
noncomputable def extendLe (hs : LinearIndependent K ((↑) : s → V))
    (hst : s ⊆ t) (ht : ⊤ ≤ span K t) :
    Basis (hs.extend hst) K V :=
  Basis.mk
    (@LinearIndependent.restrict_of_comp_subtype _ _ _ id _ _ _ _ (hs.linearIndependent_extend _))
                                              /-
                                                ι : Type u_1
                                                ι' : Type u_2
                                                K : Type u_3
                                                V : Type u_4
                                                V' : Type u_5
                                                inst✝⁴ : DivisionRing K
                                                inst✝³ : AddCommGroup V
                                                inst✝² : AddCommGroup V'
                                                inst✝¹ : Module K V
                                                inst✝ : Module K V'
                                                v : ι → V
                                                s t : Set V
                                                x y z : V
                                                hs : LinearIndependent K Subtype.val
                                                hst : HasSubset.Subset s t
                                                ht : LE.le Top.top (Submodule.span K t)
                                                ⊢ HasSubset.Subset t ↑(Submodule.span K (Set.range ((hs.extend hst).restrict i …
                                              -/
    (le_trans ht <| Submodule.span_le.2 <| by simpa using hs.subset_span_extend hst)
                                              /-
                                                🎉 no goals
                                              -/


theorem extendLe_apply_self (hs : LinearIndependent K ((↑) : s → V))
    (hst : s ⊆ t) (ht : ⊤ ≤ span K t) (x : hs.extend hst) :
    Basis.extendLe hs hst ht x = x :=
  Basis.mk_apply _ _ _


@[simp]
theorem coe_extendLe (hs : LinearIndependent K ((↑) : s → V))
    (hst : s ⊆ t) (ht : ⊤ ≤ span K t) : ⇑(Basis.extendLe hs hst ht) = ((↑) : _ → _) :=
  funext (extendLe_apply_self hs hst ht)


theorem range_extendLe (hs : LinearIndependent K ((↑) : s → V))
    (hst : s ⊆ t) (ht : ⊤ ≤ span K t) :
    range (Basis.extendLe hs hst ht) = hs.extend hst := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s t : Set V
    hs : LinearIndependent K Subtype.val
    hst : HasSubset.Subset s t
    ht : LE.le Top.top (Submodule.span K t)
    ⊢ Eq (Set.range ⇑(Basis.extendLe hs hst ht)) (hs.extend hst)
  -/
  rw [coe_extendLe, Subtype.range_coe_subtype, setOf_mem_eq]
  /-
    🎉 no goals
  -/


theorem subset_extendLe (hs : LinearIndependent K ((↑) : s → V))
    (hst : s ⊆ t) (ht : ⊤ ≤ span K t) :
    s ⊆ range (Basis.extendLe hs hst ht) :=
  (range_extendLe hs hst ht).symm ▸ hs.subset_extend hst


theorem extendLe_subset (hs : LinearIndependent K ((↑) : s → V))
    (hst : s ⊆ t) (ht : ⊤ ≤ span K t) :
    range (Basis.extendLe hs hst ht) ⊆ t :=
  (range_extendLe hs hst ht).symm ▸ hs.extend_subset hst


/-- If a set `s` spans the space, this is a basis contained in `s`. -/
noncomputable def ofSpan (hs : ⊤ ≤ span K s) :
    Basis ((linearIndependent_empty K V).extend (empty_subset s)) K V :=
  extendLe (linearIndependent_empty K V) (empty_subset s) hs


theorem ofSpan_apply_self (hs : ⊤ ≤ span K s)
    (x : (linearIndependent_empty K V).extend (empty_subset s)) :
    Basis.ofSpan hs x = x :=
  extendLe_apply_self (linearIndependent_empty K V) (empty_subset s) hs x


@[simp]
theorem coe_ofSpan (hs : ⊤ ≤ span K s) : ⇑(ofSpan hs) = ((↑) : _ → _) :=
  funext (ofSpan_apply_self hs)


theorem range_ofSpan (hs : ⊤ ≤ span K s) :
    range (ofSpan hs) = (linearIndependent_empty K V).extend (empty_subset s) := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Set V
    hs : LE.le Top.top (Submodule.span K s)
    ⊢ Eq (Set.range ⇑(Basis.ofSpan hs)) (⋯.extend ⋯)
  -/
  rw [coe_ofSpan, Subtype.range_coe_subtype, setOf_mem_eq]
  /-
    🎉 no goals
  -/


theorem ofSpan_subset (hs : ⊤ ≤ span K s) : range (ofSpan hs) ⊆ s :=
  extendLe_subset (linearIndependent_empty K V) (empty_subset s) hs


/-- A set used to index `Basis.ofVectorSpace`. -/
noncomputable def ofVectorSpaceIndex : Set V :=
  (linearIndependent_empty K V).extend (subset_univ _)


/-- Each vector space has a basis. -/
noncomputable def ofVectorSpace : Basis (ofVectorSpaceIndex K V) K V :=
  Basis.extend (linearIndependent_empty K V)


@[stacks 09FN "Generalized from fields to division rings."]
instance (priority := 100) _root_.Module.Free.of_divisionRing : Module.Free K V :=
  Module.Free.of_basis (ofVectorSpace K V)


theorem ofVectorSpace_apply_self (x : ofVectorSpaceIndex K V) : ofVectorSpace K V x = x := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((Basis.ofVectorSpace K V) x) ↑x
  -/
  unfold ofVectorSpace
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((Basis.extend ⋯) x) ↑x
  -/
  exact Basis.mk_apply _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_ofVectorSpace : ⇑(ofVectorSpace K V) = ((↑) : _ → _ ) :=
  funext fun x => ofVectorSpace_apply_self K V x


theorem ofVectorSpaceIndex.linearIndependent :
    LinearIndependent K ((↑) : ofVectorSpaceIndex K V → V) := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ LinearIndependent K Subtype.val
  -/
  convert (ofVectorSpace K V).linearIndependent
  /-
    case h.e'_4
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    ⊢ Eq Subtype.val ⇑(Basis.ofVectorSpace K V)
  -/
  ext x
  /-
    case h.e'_4.h
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : Subtype fun x => Membership.mem (Basis.ofVectorSpaceIndex K V) x
    ⊢ Eq (↑x) ((Basis.ofVectorSpace K V) x)
  -/
  rw [ofVectorSpace_apply_self]
  /-
    🎉 no goals
  -/


theorem range_ofVectorSpace : range (ofVectorSpace K V) = ofVectorSpaceIndex K V :=
  range_extend _


theorem exists_basis : ∃ s : Set V, Nonempty (Basis s K V) :=
  ⟨ofVectorSpaceIndex K V, ⟨ofVectorSpace K V⟩⟩


theorem VectorSpace.card_fintype [Fintype K] [Fintype V] : ∃ n : ℕ, card V = card K ^ n := by
  classical
  exact ⟨card (Basis.ofVectorSpaceIndex K V), Module.card_fintype (Basis.ofVectorSpace K V)⟩


/-- For a module over a division ring, the span of a nonzero element is an atom of the
lattice of submodules. -/
theorem nonzero_span_atom (v : V) (hv : v ≠ 0) : IsAtom (span K {v} : Submodule K V) := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : V
    hv : Ne v 0
    ⊢ IsAtom (Submodule.span K (Singleton.singleton v))
  -/
  constructor
    /-
      case left
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ Ne (Submodule.span K (Singleton.singleton v)) Bot.bot
    -/
  · rw [Submodule.ne_bot_iff]
    /-
      case left
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ Exists fun x => And (Membership.mem (Submodule.span K (Singleton.singleton v …
    -/
    exact ⟨v, ⟨mem_span_singleton_self v, hv⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case right
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ ∀ (b : Submodule K V), LT.lt b (Submodule.span K (Singleton.singleton v)) →  …
    -/
  · intro T hT
    /-
      case right
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      ⊢ Eq T Bot.bot
    -/
    by_contra h
    /-
      case right
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      h : Not (Eq T Bot.bot)
      ⊢ False
    -/
    apply hT.2
    /-
      case right
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      h : Not (Eq T Bot.bot)
      ⊢ HasSubset.Subset ↑(Submodule.span K (Singleton.singleton v)) ↑T
    -/
    change span K {v} ≤ T
    /-
      case right
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      h : Not (Eq T Bot.bot)
      ⊢ LE.le (Submodule.span K (Singleton.singleton v)) T
    -/
    simp_rw [span_singleton_le_iff_mem, ← Ne.eq_def, Submodule.ne_bot_iff] at *
    /-
      case right
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      h : Exists fun x => And (Membership.mem T x) (Ne x 0)
      ⊢ Membership.mem T v
    -/
    rcases h with ⟨s, ⟨hs, hz⟩⟩
    /-
      case right.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      s : V
      hs : Membership.mem T s
      hz : Ne s 0
      ⊢ Membership.mem T v
    -/
    rcases mem_span_singleton.1 (hT.1 hs) with ⟨a, rfl⟩
    /-
      case right.intro.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      T : Submodule K V
      hT : LT.lt T (Submodule.span K (Singleton.singleton v))
      a : K
      hs : Membership.mem T (HSMul.hSMul a v)
      hz : Ne (HSMul.hSMul a v) 0
      ⊢ Membership.mem T v
    -/
    rcases eq_or_ne a 0 with rfl | h
      /-
        case right.intro.intro.intro.inl
        K : Type u_3
        V : Type u_4
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        v : V
        hv : Ne v 0
        T : Submodule K V
        hT : LT.lt T (Submodule.span K (Singleton.singleton v))
        hs : Membership.mem T (HSMul.hSMul 0 v)
        hz : Ne (HSMul.hSMul 0 v) 0
        ⊢ Membership.mem T v
      -/
    · simp only [zero_smul, ne_eq, not_true] at hz
      /-
        🎉 no goals
      -/
      /-
        case right.intro.intro.intro.inr
        K : Type u_3
        V : Type u_4
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        v : V
        hv : Ne v 0
        T : Submodule K V
        hT : LT.lt T (Submodule.span K (Singleton.singleton v))
        a : K
        hs : Membership.mem T (HSMul.hSMul a v)
        hz : Ne (HSMul.hSMul a v) 0
        h : Ne a 0
        ⊢ Membership.mem T v
      -/
    · rwa [T.smul_mem_iff h] at hs
      /-
        🎉 no goals
      -/


/-- The atoms of the lattice of submodules of a module over a division ring are the
submodules equal to the span of a nonzero element of the module. -/
theorem atom_iff_nonzero_span (W : Submodule K V) :
    IsAtom W ↔ ∃ v ≠ 0, W = span K {v} := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    W : Submodule K V
    ⊢ Iff (IsAtom W) (Exists fun v => And (Ne v 0) (Eq W (Submodule.span K (Single …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      h : IsAtom W
      ⊢ Exists fun v => And (Ne v 0) (Eq W (Submodule.span K (Singleton.singleton v)))
    -/
  · cases' h with hbot h
    /-
      case refine_1.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      hbot : Ne W Bot.bot
      h : ∀ (b : Submodule K V), LT.lt b W → Eq b Bot.bot
      ⊢ Exists fun v => And (Ne v 0) (Eq W (Submodule.span K (Singleton.singleton v)))
    -/
    rcases (Submodule.ne_bot_iff W).1 hbot with ⟨v, ⟨hW, hv⟩⟩
    /-
      case refine_1.intro.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      hbot : Ne W Bot.bot
      h : ∀ (b : Submodule K V), LT.lt b W → Eq b Bot.bot
      v : V
      hW : Membership.mem W v
      hv : Ne v 0
      ⊢ Exists fun v => And (Ne v 0) (Eq W (Submodule.span K (Singleton.singleton v)))
    -/
    refine ⟨v, ⟨hv, ?_⟩⟩
    /-
      case refine_1.intro.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      hbot : Ne W Bot.bot
      h : ∀ (b : Submodule K V), LT.lt b W → Eq b Bot.bot
      v : V
      hW : Membership.mem W v
      hv : Ne v 0
      ⊢ Eq W (Submodule.span K (Singleton.singleton v))
    -/
    by_contra heq
    /-
      case refine_1.intro.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      hbot : Ne W Bot.bot
      h : ∀ (b : Submodule K V), LT.lt b W → Eq b Bot.bot
      v : V
      hW : Membership.mem W v
      hv : Ne v 0
      heq : Not (Eq W (Submodule.span K (Singleton.singleton v)))
      ⊢ False
    -/
    specialize h (span K {v})
    /-
      case refine_1.intro.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      hbot : Ne W Bot.bot
      v : V
      hW : Membership.mem W v
      hv : Ne v 0
      heq : Not (Eq W (Submodule.span K (Singleton.singleton v)))
      h : LT.lt (Submodule.span K (Singleton.singleton v)) W → Eq (Submodule.span K  …
      ⊢ False
    -/
    rw [span_singleton_eq_bot, lt_iff_le_and_ne] at h
    /-
      case refine_1.intro.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      hbot : Ne W Bot.bot
      v : V
      hW : Membership.mem W v
      hv : Ne v 0
      heq : Not (Eq W (Submodule.span K (Singleton.singleton v)))
      h : And (LE.le (Submodule.span K (Singleton.singleton v)) W) (Ne (Submodule.sp …
      ⊢ False
    -/
    exact hv (h ⟨(span_singleton_le_iff_mem v W).2 hW, Ne.symm heq⟩)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      W : Submodule K V
      h : Exists fun v => And (Ne v 0) (Eq W (Submodule.span K (Singleton.singleton  …
      ⊢ IsAtom W
    -/
  · rcases h with ⟨v, ⟨hv, rfl⟩⟩
    /-
      case refine_2.intro.intro
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ IsAtom (Submodule.span K (Singleton.singleton v))
    -/
    exact nonzero_span_atom v hv
    /-
      🎉 no goals
    -/


/-- The lattice of submodules of a module over a division ring is atomistic. -/
instance : IsAtomistic (Submodule K V) where
  eq_sSup_atoms W := by
    /-
      ι : Type u_1
      ι' : Type u_2
      K : Type u_3
      V : Type u_4
      V' : Type u_5
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup V'
      inst✝¹ : Module K V
      inst✝ : Module K V'
      v : ι → V
      s t : Set V
      x y z : V
      W : Submodule K V
      ⊢ Exists fun s => And (Eq W (SupSet.sSup s)) (∀ (a : Submodule K V), Membershi …
    -/
    refine ⟨_, submodule_eq_sSup_le_nonzero_spans W, ?_⟩
    /-
      ι : Type u_1
      ι' : Type u_2
      K : Type u_3
      V : Type u_4
      V' : Type u_5
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup V'
      inst✝¹ : Module K V
      inst✝ : Module K V'
      v : ι → V
      s t : Set V
      x y z : V
      W : Submodule K V
      ⊢ ∀ (a : Submodule K V), Membership.mem (setOf fun T => Exists fun m => And (M …
    -/
    rintro _ ⟨w, ⟨_, ⟨hw, rfl⟩⟩⟩
    /-
      case intro.intro.intro
      ι : Type u_1
      ι' : Type u_2
      K : Type u_3
      V : Type u_4
      V' : Type u_5
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : AddCommGroup V'
      inst✝¹ : Module K V
      inst✝ : Module K V'
      v : ι → V
      s t : Set V
      x y z : V
      W : Submodule K V
      w : V
      left✝ : Membership.mem W w
      hw : Ne w 0
      ⊢ IsAtom (Submodule.span K (Singleton.singleton w))
    -/
    exact nonzero_span_atom w hw
    /-
      🎉 no goals
    -/


theorem LinearMap.exists_leftInverse_of_injective (f : V →ₗ[K] V') (hf_inj : LinearMap.ker f = ⊥) :
    ∃ g : V' →ₗ[K] V, g.comp f = LinearMap.id := by
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  let B := Basis.ofVectorSpaceIndex K V
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  let hB := Basis.ofVectorSpace K V
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  have hB₀ : _ := hB.linearIndependent.to_subtype_range
  have : LinearIndependent K (fun x => x : f '' B → V') := by
    have h₁ : LinearIndependent K ((↑) : ↥(f '' Set.range (Basis.ofVectorSpace K V)) → V') :=
      LinearIndependent.image_subtype (f := f) hB₀ (show Disjoint _ _ by simp [hf_inj])
    rwa [Basis.range_ofVectorSpace K V] at h₁
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  let C := this.extend (subset_univ _)
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  have BC := this.subset_extend (subset_univ _)
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset (Set.image (⇑f) B) (this.extend ⋯)
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  let hC := Basis.extend this
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset (Set.image (⇑f) B) (this.extend ⋯)
    hC : Basis (↑(this.extend ⋯)) K V' := Basis.extend this
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  haveI Vinh : Inhabited V := ⟨0⟩
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset (Set.image (⇑f) B) (this.extend ⋯)
    hC : Basis (↑(this.extend ⋯)) K V' := Basis.extend this
    Vinh : Inhabited V
    ⊢ Exists fun g => Eq (g.comp f) LinearMap.id
  -/
  refine ⟨(hC.constr ℕ : _ → _) (C.restrict (invFun f)), hB.ext fun b => ?_⟩
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset (Set.image (⇑f) B) (this.extend ⋯)
    hC : Basis (↑(this.extend ⋯)) K V' := Basis.extend this
    Vinh : Inhabited V
    b : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((((hC.constr Nat) (C.restrict (Function.invFun ⇑f))).comp f) (hB b)) (Li …
  -/
  rw [image_subset_iff] at BC
  have fb_eq : f b = hC ⟨f b, BC b.2⟩ := by
    change f b = Basis.extend this _
    simp_rw [Basis.extend_apply_self]
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset B (Set.preimage (⇑f) (this.extend ⋯))
    hC : Basis (↑(this.extend ⋯)) K V' := Basis.extend this
    Vinh : Inhabited V
    b : ↑(Basis.ofVectorSpaceIndex K V)
    fb_eq : Eq (f ↑b) (hC ⟨f ↑b, ⋯⟩)
    ⊢ Eq ((((hC.constr Nat) (C.restrict (Function.invFun ⇑f))).comp f) (hB b)) (Li …
  -/
  dsimp []
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset B (Set.preimage (⇑f) (this.extend ⋯))
    hC : Basis (↑(this.extend ⋯)) K V' := Basis.extend this
    Vinh : Inhabited V
    b : ↑(Basis.ofVectorSpaceIndex K V)
    fb_eq : Eq (f ↑b) (hC ⟨f ↑b, ⋯⟩)
    ⊢ Eq (((hC.constr Nat) (C.restrict (Function.invFun ⇑f))) (f (hB b))) (hB b)
  -/
  rw [Basis.ofVectorSpace_apply_self, fb_eq, hC.constr_basis]
  /-
    K : Type u_3
    V : Type u_4
    V' : Type u_5
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V
    inst✝ : Module K V'
    f : LinearMap (RingHom.id K) V V'
    hf_inj : Eq (LinearMap.ker f) Bot.bot
    B : Set V := Basis.ofVectorSpaceIndex K V
    hB : Basis (↑(Basis.ofVectorSpaceIndex K V)) K V := Basis.ofVectorSpace K V
    hB₀ : LinearIndependent K Subtype.val
    this : LinearIndependent K fun x => ↑x
    C : Set V' := this.extend ⋯
    BC : HasSubset.Subset B (Set.preimage (⇑f) (this.extend ⋯))
    hC : Basis (↑(this.extend ⋯)) K V' := Basis.extend this
    Vinh : Inhabited V
    b : ↑(Basis.ofVectorSpaceIndex K V)
    fb_eq : Eq (f ↑b) (hC ⟨f ↑b, ⋯⟩)
    ⊢ Eq (C.restrict (Function.invFun ⇑f) ⟨f ↑b, ⋯⟩) ↑b
  -/
  exact leftInverse_invFun (LinearMap.ker_eq_bot.1 hf_inj) _
  /-
    🎉 no goals
  -/


theorem Submodule.exists_isCompl (p : Submodule K V) : ∃ q : Submodule K V, IsCompl p q :=
  let ⟨f, hf⟩ := p.subtype.exists_leftInverse_of_injective p.ker_subtype
  ⟨LinearMap.ker f, LinearMap.isCompl_of_proj <| LinearMap.ext_iff.1 hf⟩


instance Submodule.complementedLattice : ComplementedLattice (Submodule K V) :=
  ⟨Submodule.exists_isCompl⟩


/-- Any linear map `f : p →ₗ[K] V'` defined on a subspace `p` can be extended to the whole
space. -/
theorem LinearMap.exists_extend {p : Submodule K V} (f : p →ₗ[K] V') :
    ∃ g : V →ₗ[K] V', g.comp p.subtype = f :=
  let ⟨g, hg⟩ := p.subtype.exists_leftInverse_of_injective p.ker_subtype
                /-
                  K : Type u_3
                  V : Type u_4
                  V' : Type u_5
                  inst✝⁴ : DivisionRing K
                  inst✝³ : AddCommGroup V
                  inst✝² : AddCommGroup V'
                  inst✝¹ : Module K V
                  inst✝ : Module K V'
                  p : Submodule K V
                  f : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem p x) V'
                  g : LinearMap (RingHom.id K) V (Subtype fun x => Membership.mem p x)
                  hg : Eq (g.comp p.subtype) LinearMap.id
                  ⊢ Eq ((f.comp g).comp p.subtype) f
                -/
  ⟨f.comp g, by rw [LinearMap.comp_assoc, hg, f.comp_id]⟩
                /-
                  🎉 no goals
                -/


/-- If `p < ⊤` is a subspace of a vector space `V`, then there exists a nonzero linear map
`f : V →ₗ[K] K` such that `p ≤ ker f`. -/
theorem Submodule.exists_le_ker_of_lt_top (p : Submodule K V) (hp : p < ⊤) :
    ∃ (f : V →ₗ[K] K), f ≠ 0 ∧ p ≤ ker f := by
  /-
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    p : Submodule K V
    hp : LT.lt p Top.top
    ⊢ Exists fun f => And (Ne f 0) (LE.le p (LinearMap.ker f))
  -/
  rcases SetLike.exists_of_lt hp with ⟨v, -, hpv⟩; clear hp
  /-
    case intro.intro
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    p : Submodule K V
    v : V
    hpv : Not (Membership.mem p v)
    ⊢ Exists fun f => And (Ne f 0) (LE.le p (LinearMap.ker f))
  -/
  rcases (LinearPMap.supSpanSingleton ⟨p, 0⟩ v (1 : K) hpv).toFun.exists_extend with ⟨f, hf⟩
  /-
    case intro.intro.intro
    K : Type u_3
    V : Type u_4
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    p : Submodule K V
    v : V
    hpv : Not (Membership.mem p v)
    f : LinearMap (RingHom.id K) V K
    hf : Eq (f.comp ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).domain. …
    ⊢ Exists fun f => And (Ne f 0) (LE.le p (LinearMap.ker f))
  -/
  refine ⟨f, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      f : LinearMap (RingHom.id K) V K
      hf : Eq (f.comp ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).domain. …
      ⊢ Ne f 0
    -/
  · rintro rfl
    /-
      case intro.intro.intro.refine_1
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      hf : Eq (LinearMap.comp 0 ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hp …
      ⊢ False
    -/
    rw [LinearMap.zero_comp] at hf
    /-
      case intro.intro.intro.refine_1
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      hf : Eq 0 ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).toFun
      ⊢ False
    -/
    have := LinearPMap.supSpanSingleton_apply_mk ⟨p, 0⟩ v (1 : K) hpv 0 p.zero_mem 1
    /-
      case intro.intro.intro.refine_1
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      hf : Eq 0 ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).toFun
      this : Eq (↑({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv) ⟨HAdd.hAdd  …
      ⊢ False
    -/
    simpa using (LinearMap.congr_fun hf _).trans this
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      f : LinearMap (RingHom.id K) V K
      hf : Eq (f.comp ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).domain. …
      ⊢ LE.le p (LinearMap.ker f)
    -/
  · refine fun x hx => mem_ker.2 ?_
    /-
      case intro.intro.intro.refine_2
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      f : LinearMap (RingHom.id K) V K
      hf : Eq (f.comp ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).domain. …
      x : V
      hx : Membership.mem p x
      ⊢ Eq (f x) 0
    -/
    have := LinearPMap.supSpanSingleton_apply_mk ⟨p, 0⟩ v (1 : K) hpv x hx 0
    /-
      case intro.intro.intro.refine_2
      K : Type u_3
      V : Type u_4
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      p : Submodule K V
      v : V
      hpv : Not (Membership.mem p v)
      f : LinearMap (RingHom.id K) V K
      hf : Eq (f.comp ({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv).domain. …
      x : V
      hx : Membership.mem p x
      this : Eq (↑({ domain := p, toFun := 0 }.supSpanSingleton v 1 hpv) ⟨HAdd.hAdd  …
      ⊢ Eq (f x) 0
    -/
    simpa using (LinearMap.congr_fun hf _).trans this
    /-
      🎉 no goals
    -/


theorem quotient_prod_linearEquiv (p : Submodule K V) : Nonempty (((V ⧸ p) × p) ≃ₗ[K] V) :=
  let ⟨q, hq⟩ := p.exists_isCompl
  Nonempty.intro <|
    ((quotientEquivOfIsCompl p q hq).prod (LinearEquiv.refl _ _)).trans
      (prodEquivOfIsCompl q p hq.symm)


