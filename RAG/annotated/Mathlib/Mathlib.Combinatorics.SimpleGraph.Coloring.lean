/-- An `α`-coloring of a simple graph `G` is a homomorphism of `G` into the complete graph on `α`.
This is also known as a proper coloring.
-/
abbrev Coloring (α : Type v) := G →g (⊤ : SimpleGraph α)


theorem Coloring.valid {v w : V} (h : G.Adj v w) : C v ≠ C w :=
  C.map_rel h


/-- Construct a term of `SimpleGraph.Coloring` using a function that
assigns vertices to colors and a proof that it is as proper coloring.

(Note: this is a definitionally the constructor for `SimpleGraph.Hom`,
but with a syntactically better proper coloring hypothesis.)
-/
@[match_pattern]
def Coloring.mk (color : V → α) (valid : ∀ {v w : V}, G.Adj v w → color v ≠ color w) :
    G.Coloring α :=
  ⟨color, @valid⟩


/-- The color class of a given color.
-/
def Coloring.colorClass (c : α) : Set V := { v : V | C v = c }


/-- The set containing all color classes. -/
def Coloring.colorClasses : Set (Set V) := (Setoid.ker C).classes


theorem Coloring.mem_colorClass (v : V) : v ∈ C.colorClass (C v) := rfl


theorem Coloring.colorClasses_isPartition : Setoid.IsPartition C.colorClasses :=
  Setoid.isPartition_classes (Setoid.ker C)


theorem Coloring.mem_colorClasses {v : V} : C.colorClass (C v) ∈ C.colorClasses :=
  ⟨v, rfl⟩


theorem Coloring.colorClasses_finite [Finite α] : C.colorClasses.Finite :=
  Setoid.finite_classes_ker _


theorem Coloring.card_colorClasses_le [Fintype α] [Fintype C.colorClasses] :
    Fintype.card C.colorClasses ≤ Fintype.card α := by
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    C : G.Coloring α
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑C.colorClasses
    ⊢ LE.le (Fintype.card ↑C.colorClasses) (Fintype.card α)
  -/
  simp only [colorClasses]
  -- Porting note: brute force instance declaration `[Fintype (Setoid.classes (Setoid.ker C))]`
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    C : G.Coloring α
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑C.colorClasses
    ⊢ LE.le (Fintype.card ↑(Setoid.ker ⇑C).classes) (Fintype.card α)
  -/
  haveI : Fintype (Setoid.classes (Setoid.ker C)) := by assumption
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    C : G.Coloring α
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑C.colorClasses
    this : Fintype ↑(Setoid.ker ⇑C).classes
    ⊢ LE.le (Fintype.card ↑(Setoid.ker ⇑C).classes) (Fintype.card α)
  -/
  convert Setoid.card_classes_ker_le C
  /-
    🎉 no goals
  -/


theorem Coloring.not_adj_of_mem_colorClass {c : α} {v w : V} (hv : v ∈ C.colorClass c)
    (hw : w ∈ C.colorClass c) : ¬G.Adj v w := fun h => C.valid h (Eq.trans hv (Eq.symm hw))


theorem Coloring.color_classes_independent (c : α) : IsAntichain G.Adj (C.colorClass c) :=
  fun _ hv _ hw _ => C.not_adj_of_mem_colorClass hv hw

-- TODO make this computable

noncomputable instance [Fintype V] [Fintype α] : Fintype (Coloring G α) := by
  classical
  change Fintype (RelHom G.Adj (⊤ : SimpleGraph α).Adj)
  apply Fintype.ofInjective _ RelHom.coe_fn_injective


/-- Whether a graph can be colored by at most `n` colors. -/
def Colorable (n : ℕ) : Prop := Nonempty (G.Coloring (Fin n))


/-- The coloring of an empty graph. -/
def coloringOfIsEmpty [IsEmpty V] : G.Coloring α :=
  Coloring.mk isEmptyElim fun {v} => isEmptyElim v


theorem colorable_of_isEmpty [IsEmpty V] (n : ℕ) : G.Colorable n :=
  ⟨G.coloringOfIsEmpty⟩


theorem isEmpty_of_colorable_zero (h : G.Colorable 0) : IsEmpty V := by
  /-
    V : Type u
    G : SimpleGraph V
    h : G.Colorable 0
    ⊢ IsEmpty V
  -/
  constructor
  /-
    case false
    V : Type u
    G : SimpleGraph V
    h : G.Colorable 0
    ⊢ V → False
  -/
  intro v
  /-
    case false
    V : Type u
    G : SimpleGraph V
    h : G.Colorable 0
    v : V
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := h.some v
  /-
    case false.mk
    V : Type u
    G : SimpleGraph V
    h : G.Colorable 0
    v : V
    i : Nat
    hi : LT.lt i 0
    ⊢ False
  -/
  exact Nat.not_lt_zero _ hi
  /-
    🎉 no goals
  -/


/-- The "tautological" coloring of a graph, using the vertices of the graph as colors. -/
def selfColoring : G.Coloring V := Coloring.mk id fun {_ _} => G.ne_of_adj


/-- The chromatic number of a graph is the minimal number of colors needed to color it.
This is `⊤` (infinity) iff `G` isn't colorable with finitely many colors.

If `G` is colorable, then `ENat.toNat G.chromaticNumber` is the `ℕ`-valued chromatic number. -/
noncomputable def chromaticNumber : ℕ∞ := ⨅ n ∈ setOf G.Colorable, (n : ℕ∞)


lemma chromaticNumber_eq_biInf {G : SimpleGraph V} :
    G.chromaticNumber = ⨅ n ∈ setOf G.Colorable, (n : ℕ∞) := rfl


lemma chromaticNumber_eq_iInf {G : SimpleGraph V} :
    G.chromaticNumber = ⨅ n : {m | G.Colorable m}, (n : ℕ∞) := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Eq G.chromaticNumber (iInf fun n => ↑↑n)
  -/
  rw [chromaticNumber, iInf_subtype]
  /-
    🎉 no goals
  -/


lemma Colorable.chromaticNumber_eq_sInf {G : SimpleGraph V} {n} (h : G.Colorable n) :
    G.chromaticNumber = sInf {n' : ℕ | G.Colorable n'} := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : G.Colorable n
    ⊢ Eq G.chromaticNumber ↑(InfSet.sInf (setOf fun n' => G.Colorable n'))
  -/
  rw [ENat.coe_sInf, chromaticNumber]
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : G.Colorable n
    ⊢ (setOf fun n' => G.Colorable n').Nonempty
  -/
  exact ⟨_, h⟩
  /-
    🎉 no goals
  -/


/-- Given an embedding, there is an induced embedding of colorings. -/
def recolorOfEmbedding {α β : Type*} (f : α ↪ β) : G.Coloring α ↪ G.Coloring β where
  toFun C := (Embedding.completeGraph f).toHom.comp C
  inj' := by -- this was strangely painful; seems like missing lemmas about embeddings
    /-
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      ⊢ Function.Injective fun C => (SimpleGraph.Embedding.completeGraph f).toHom.co …
    -/
    intro C C' h
    /-
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      C C' : G.Coloring α
      h : Eq ((fun C => (SimpleGraph.Embedding.completeGraph f).toHom.comp C) C) ((f …
      ⊢ Eq C C'
    -/
    dsimp only at h
    /-
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      C C' : G.Coloring α
      h : Eq ((SimpleGraph.Embedding.completeGraph f).toHom.comp C) ((SimpleGraph.Em …
      ⊢ Eq C C'
    -/
    ext v
    /-
      case h
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      C C' : G.Coloring α
      h : Eq ((SimpleGraph.Embedding.completeGraph f).toHom.comp C) ((SimpleGraph.Em …
      v : V
      ⊢ Eq (C v) (C' v)
    -/
    apply (Embedding.completeGraph f).inj'
    /-
      case h.a
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      C C' : G.Coloring α
      h : Eq ((SimpleGraph.Embedding.completeGraph f).toHom.comp C) ((SimpleGraph.Em …
      v : V
      ⊢ Eq ((SimpleGraph.Embedding.completeGraph f).toFun (C v)) ((SimpleGraph.Embed …
    -/
    change ((Embedding.completeGraph f).toHom.comp C) v = _
    /-
      case h.a
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      C C' : G.Coloring α
      h : Eq ((SimpleGraph.Embedding.completeGraph f).toHom.comp C) ((SimpleGraph.Em …
      v : V
      ⊢ Eq (((SimpleGraph.Embedding.completeGraph f).toHom.comp C) v) ((SimpleGraph. …
    -/
    rw [h]
    /-
      case h.a
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Function.Embedding α β
      C C' : G.Coloring α
      h : Eq ((SimpleGraph.Embedding.completeGraph f).toHom.comp C) ((SimpleGraph.Em …
      v : V
      ⊢ Eq (((SimpleGraph.Embedding.completeGraph f).toHom.comp C') v) ((SimpleGraph …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp] lemma coe_recolorOfEmbedding (f : α ↪ β) :
    ⇑(G.recolorOfEmbedding f) = (Embedding.completeGraph f).toHom.comp := rfl


/-- Given an equivalence, there is an induced equivalence between colorings. -/
def recolorOfEquiv {α β : Type*} (f : α ≃ β) : G.Coloring α ≃ G.Coloring β where
  toFun := G.recolorOfEmbedding f.toEmbedding
  invFun := G.recolorOfEmbedding f.symm.toEmbedding
  left_inv C := by
    /-
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Equiv α β
      C : G.Coloring α
      ⊢ Eq ((G.recolorOfEmbedding f.symm.toEmbedding) ((G.recolorOfEmbedding f.toEmb …
    -/
    ext v
    /-
      case h
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Equiv α β
      C : G.Coloring α
      v : V
      ⊢ Eq (((G.recolorOfEmbedding f.symm.toEmbedding) ((G.recolorOfEmbedding f.toEm …
    -/
    apply Equiv.symm_apply_apply
    /-
      🎉 no goals
    -/
  right_inv C := by
    /-
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Equiv α β
      C : G.Coloring β
      ⊢ Eq ((G.recolorOfEmbedding f.toEmbedding) ((G.recolorOfEmbedding f.symm.toEmb …
    -/
    ext v
    /-
      case h
      V : Type u
      G : SimpleGraph V
      n : Nat
      α✝ : Type u_1
      β✝ : Type u_2
      C✝ : G.Coloring α✝
      α : Type u_3
      β : Type u_4
      f : Equiv α β
      C : G.Coloring β
      v : V
      ⊢ Eq (((G.recolorOfEmbedding f.toEmbedding) ((G.recolorOfEmbedding f.symm.toEm …
    -/
    apply Equiv.apply_symm_apply
    /-
      🎉 no goals
    -/


@[simp] lemma coe_recolorOfEquiv (f : α ≃ β) :
    ⇑(G.recolorOfEquiv f) = (Embedding.completeGraph f).toHom.comp := rfl


/-- There is a noncomputable embedding of `α`-colorings to `β`-colorings if
`β` has at least as large a cardinality as `α`. -/
noncomputable def recolorOfCardLE {α β : Type*} [Fintype α] [Fintype β]
    (hn : Fintype.card α ≤ Fintype.card β) : G.Coloring α ↪ G.Coloring β :=
  G.recolorOfEmbedding <| (Function.Embedding.nonempty_of_card_le hn).some


@[simp] lemma coe_recolorOfCardLE [Fintype α] [Fintype β] (hαβ : card α ≤ card β) :
    ⇑(G.recolorOfCardLE hαβ) =
      (Embedding.completeGraph (Embedding.nonempty_of_card_le hαβ).some).toHom.comp := rfl


theorem Colorable.mono {n m : ℕ} (h : n ≤ m) (hc : G.Colorable n) : G.Colorable m :=
                         /-
                           V : Type u
                           G : SimpleGraph V
                           n m : Nat
                           h : LE.le n m
                           hc : G.Colorable n
                           ⊢ LE.le (Fintype.card (Fin n)) (Fintype.card (Fin m))
                         -/
  ⟨G.recolorOfCardLE (by simp [h]) hc.some⟩
                         /-
                           🎉 no goals
                         -/


theorem Coloring.colorable [Fintype α] (C : G.Coloring α) : G.Colorable (Fintype.card α) :=
                         /-
                           V : Type u
                           G : SimpleGraph V
                           α : Type u_1
                           inst✝ : Fintype α
                           C : G.Coloring α
                           ⊢ LE.le (Fintype.card α) (Fintype.card (Fin (Fintype.card α)))
                         -/
  ⟨G.recolorOfCardLE (by simp) C⟩
                         /-
                           🎉 no goals
                         -/


theorem colorable_of_fintype (G : SimpleGraph V) [Fintype V] : G.Colorable (Fintype.card V) :=
  G.selfColoring.colorable


/-- Noncomputably get a coloring from colorability. -/
noncomputable def Colorable.toColoring [Fintype α] {n : ℕ} (hc : G.Colorable n)
    (hn : n ≤ Fintype.card α) : G.Coloring α := by
  /-
    V : Type u
    G : SimpleGraph V
    n✝ : Nat
    α : Type u_1
    β : Type u_2
    C : G.Coloring α
    inst✝ : Fintype α
    n : Nat
    hc : G.Colorable n
    hn : LE.le n (Fintype.card α)
    ⊢ G.Coloring α
  -/
  rw [← Fintype.card_fin n] at hn
  /-
    V : Type u
    G : SimpleGraph V
    n✝ : Nat
    α : Type u_1
    β : Type u_2
    C : G.Coloring α
    inst✝ : Fintype α
    n : Nat
    hc : G.Colorable n
    hn : LE.le (Fintype.card (Fin n)) (Fintype.card α)
    ⊢ G.Coloring α
  -/
  exact G.recolorOfCardLE hn hc.some
  /-
    🎉 no goals
  -/


theorem Colorable.of_embedding {V' : Type*} {G' : SimpleGraph V'} (f : G ↪g G') {n : ℕ}
    (h : G'.Colorable n) : G.Colorable n :=
                     /-
                       V : Type u
                       G : SimpleGraph V
                       V' : Type u_3
                       G' : SimpleGraph V'
                       f : G.Embedding G'
                       n : Nat
                       h : G'.Colorable n
                       ⊢ LE.le n (Fintype.card (Fin n))
                     -/
  ⟨(h.toColoring (by simp)).comp f⟩
                     /-
                       🎉 no goals
                     -/


theorem colorable_iff_exists_bdd_nat_coloring (n : ℕ) :
    G.Colorable n ↔ ∃ C : G.Coloring ℕ, ∀ v, C v < n := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    ⊢ Iff (G.Colorable n) (Exists fun C => ∀ (v : V), LT.lt (C v) n)
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      n : Nat
      ⊢ G.Colorable n → Exists fun C => ∀ (v : V), LT.lt (C v) n
    -/
  · rintro hc
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      n : Nat
      hc : G.Colorable n
      ⊢ Exists fun C => ∀ (v : V), LT.lt (C v) n
    -/
    have C : G.Coloring (Fin n) := hc.toColoring (by simp)
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      n : Nat
      hc : G.Colorable n
      C : G.Coloring (Fin n)
      ⊢ Exists fun C => ∀ (v : V), LT.lt (C v) n
    -/
    let f := Embedding.completeGraph (@Fin.valEmbedding n)
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      n : Nat
      hc : G.Colorable n
      C : G.Coloring (Fin n)
      f : Top.top.Embedding Top.top := SimpleGraph.Embedding.completeGraph Fin.valEm …
      ⊢ Exists fun C => ∀ (v : V), LT.lt (C v) n
    -/
    use f.toHom.comp C
    /-
      case h
      V : Type u
      G : SimpleGraph V
      n : Nat
      hc : G.Colorable n
      C : G.Coloring (Fin n)
      f : Top.top.Embedding Top.top := SimpleGraph.Embedding.completeGraph Fin.valEm …
      ⊢ ∀ (v : V), LT.lt ((f.toHom.comp C) v) n
    -/
    intro v
    /-
      case h
      V : Type u
      G : SimpleGraph V
      n : Nat
      hc : G.Colorable n
      C : G.Coloring (Fin n)
      f : Top.top.Embedding Top.top := SimpleGraph.Embedding.completeGraph Fin.valEm …
      v : V
      ⊢ LT.lt ((f.toHom.comp C) v) n
    -/
    cases' C with color valid
    /-
      case h.mk
      V : Type u
      G : SimpleGraph V
      n : Nat
      hc : G.Colorable n
      f : Top.top.Embedding Top.top := SimpleGraph.Embedding.completeGraph Fin.valEm …
      v : V
      color : V → Fin n
      valid : ∀ {a b : V}, G.Adj a b → Top.top.Adj (color a) (color b)
      ⊢ LT.lt ((f.toHom.comp { toFun := color, map_rel' := valid }) v) n
    -/
    exact Fin.is_lt (color v)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      n : Nat
      ⊢ (Exists fun C => ∀ (v : V), LT.lt (C v) n) → G.Colorable n
    -/
  · rintro ⟨C, Cf⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring Nat
      Cf : ∀ (v : V), LT.lt (C v) n
      ⊢ G.Colorable n
    -/
    refine ⟨Coloring.mk ?_ ?_⟩
      /-
        case mpr.intro.refine_1
        V : Type u
        G : SimpleGraph V
        n : Nat
        C : G.Coloring Nat
        Cf : ∀ (v : V), LT.lt (C v) n
        ⊢ V → Fin n
      -/
    · exact fun v => ⟨C v, Cf v⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.refine_2
        V : Type u
        G : SimpleGraph V
        n : Nat
        C : G.Coloring Nat
        Cf : ∀ (v : V), LT.lt (C v) n
        ⊢ ∀ {v w : V}, G.Adj v w → Ne ⟨C v, ⋯⟩ ⟨C w, ⋯⟩
      -/
    · rintro v w hvw
      /-
        case mpr.intro.refine_2
        V : Type u
        G : SimpleGraph V
        n : Nat
        C : G.Coloring Nat
        Cf : ∀ (v : V), LT.lt (C v) n
        v w : V
        hvw : G.Adj v w
        ⊢ Ne ⟨C v, ⋯⟩ ⟨C w, ⋯⟩
      -/
      simp only [Fin.mk_eq_mk, Ne]
      /-
        case mpr.intro.refine_2
        V : Type u
        G : SimpleGraph V
        n : Nat
        C : G.Coloring Nat
        Cf : ∀ (v : V), LT.lt (C v) n
        v w : V
        hvw : G.Adj v w
        ⊢ Not (Eq (C v) (C w))
      -/
      exact C.valid hvw
      /-
        🎉 no goals
      -/


theorem colorable_set_nonempty_of_colorable {n : ℕ} (hc : G.Colorable n) :
    { n : ℕ | G.Colorable n }.Nonempty :=
  ⟨n, hc⟩


theorem chromaticNumber_bddBelow : BddBelow { n : ℕ | G.Colorable n } :=
  ⟨0, fun _ _ => zero_le _⟩


theorem Colorable.chromaticNumber_le {n : ℕ} (hc : G.Colorable n) : G.chromaticNumber ≤ n := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : G.Colorable n
    ⊢ LE.le G.chromaticNumber ↑n
  -/
  rw [hc.chromaticNumber_eq_sInf]
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : G.Colorable n
    ⊢ LE.le ↑(InfSet.sInf (setOf fun n' => G.Colorable n')) ↑n
  -/
  norm_cast
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : G.Colorable n
    ⊢ LE.le (InfSet.sInf (setOf fun n' => G.Colorable n')) n
  -/
  apply csInf_le chromaticNumber_bddBelow
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : G.Colorable n
    ⊢ Membership.mem (setOf fun n => G.Colorable n) n
  -/
  exact hc
  /-
    🎉 no goals
  -/


theorem chromaticNumber_ne_top_iff_exists : G.chromaticNumber ≠ ⊤ ↔ ∃ n, G.Colorable n := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff (Ne G.chromaticNumber Top.top) (Exists fun n => G.Colorable n)
  -/
  rw [chromaticNumber]
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff (Ne (iInf fun n => iInf fun h => ↑n) Top.top) (Exists fun n => G.Colorab …
  -/
  convert_to ⨅ n : {m | G.Colorable m}, (n : ℕ∞) ≠ ⊤ ↔ _
    /-
      case h.e'_1.a
      V : Type u
      G : SimpleGraph V
      ⊢ Iff (Ne (iInf fun n => iInf fun h => ↑n) Top.top) (Ne (iInf fun n => ↑↑n) To …
    -/
  · rw [iInf_subtype]
    /-
      🎉 no goals
    -/
  /-
    case convert_2
    V : Type u
    G : SimpleGraph V
    ⊢ Iff (Ne (iInf fun n => ↑↑n) Top.top) (Exists fun n => G.Colorable n)
  -/
  rw [← lt_top_iff_ne_top, ENat.iInf_coe_lt_top]
  /-
    case convert_2
    V : Type u
    G : SimpleGraph V
    ⊢ Iff (Nonempty ↑(setOf fun m => G.Colorable m)) (Exists fun n => G.Colorable n)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem chromaticNumber_le_iff_colorable {n : ℕ} : G.chromaticNumber ≤ n ↔ G.Colorable n := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    ⊢ Iff (LE.le G.chromaticNumber ↑n) (G.Colorable n)
  -/
  refine ⟨fun h ↦ ?_, Colorable.chromaticNumber_le⟩
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le G.chromaticNumber ↑n
    ⊢ G.Colorable n
  -/
  have : G.chromaticNumber ≠ ⊤ := (trans h (WithTop.coe_lt_top n)).ne
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le G.chromaticNumber ↑n
    this : Ne G.chromaticNumber Top.top
    ⊢ G.Colorable n
  -/
  rw [chromaticNumber_ne_top_iff_exists] at this
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le G.chromaticNumber ↑n
    this : Exists fun n => G.Colorable n
    ⊢ G.Colorable n
  -/
  obtain ⟨m, hm⟩ := this
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le G.chromaticNumber ↑n
    m : Nat
    hm : G.Colorable m
    ⊢ G.Colorable n
  -/
  rw [hm.chromaticNumber_eq_sInf, Nat.cast_le] at h
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le (InfSet.sInf (setOf fun n' => G.Colorable n')) n
    m : Nat
    hm : G.Colorable m
    ⊢ G.Colorable n
  -/
  have := Nat.sInf_mem (⟨m, hm⟩ : {n' | G.Colorable n'}.Nonempty)
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le (InfSet.sInf (setOf fun n' => G.Colorable n')) n
    m : Nat
    hm : G.Colorable m
    this : Membership.mem (setOf fun n' => G.Colorable n') (InfSet.sInf (setOf fun …
    ⊢ G.Colorable n
  -/
  rw [Set.mem_setOf_eq] at this
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    h : LE.le (InfSet.sInf (setOf fun n' => G.Colorable n')) n
    m : Nat
    hm : G.Colorable m
    this : G.Colorable (InfSet.sInf (setOf fun n' => G.Colorable n'))
    ⊢ G.Colorable n
  -/
  exact this.mono h
  /-
    🎉 no goals
  -/


@[deprecated Colorable.chromaticNumber_le (since := "2024-03-21")]
theorem chromaticNumber_le_card [Fintype α] (C : G.Coloring α) :
    G.chromaticNumber ≤ Fintype.card α := C.colorable.chromaticNumber_le


theorem colorable_chromaticNumber {m : ℕ} (hc : G.Colorable m) :
    G.Colorable (ENat.toNat G.chromaticNumber) := by
  classical
  rw [hc.chromaticNumber_eq_sInf, Nat.sInf_def]
  · apply Nat.find_spec
  · exact colorable_set_nonempty_of_colorable hc


theorem colorable_chromaticNumber_of_fintype (G : SimpleGraph V) [Finite V] :
    G.Colorable (ENat.toNat G.chromaticNumber) := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Finite V
    ⊢ G.Colorable G.chromaticNumber.toNat
  -/
  cases nonempty_fintype V
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    inst✝ : Finite V
    val✝ : Fintype V
    ⊢ G.Colorable G.chromaticNumber.toNat
  -/
  exact colorable_chromaticNumber G.colorable_of_fintype
  /-
    🎉 no goals
  -/


theorem chromaticNumber_le_one_of_subsingleton (G : SimpleGraph V) [Subsingleton V] :
    G.chromaticNumber ≤ 1 := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Subsingleton V
    ⊢ LE.le G.chromaticNumber 1
  -/
  rw [← Nat.cast_one, chromaticNumber_le_iff_colorable]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Subsingleton V
    ⊢ G.Colorable 1
  -/
  refine ⟨Coloring.mk (fun _ => 0) ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Subsingleton V
    ⊢ ∀ {v w : V}, G.Adj v w → Ne ((fun x => 0) v) ((fun x => 0) w)
  -/
  intros v w
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Subsingleton V
    v w : V
    ⊢ G.Adj v w → Ne ((fun x => 0) v) ((fun x => 0) w)
  -/
  cases Subsingleton.elim v w
  /-
    case refl
    V : Type u
    G : SimpleGraph V
    inst✝ : Subsingleton V
    v : V
    ⊢ G.Adj v v → Ne ((fun x => 0) v) ((fun x => 0) v)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem chromaticNumber_eq_zero_of_isempty (G : SimpleGraph V) [IsEmpty V] :
    G.chromaticNumber = 0 := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : IsEmpty V
    ⊢ Eq G.chromaticNumber 0
  -/
  rw [← nonpos_iff_eq_zero, ← Nat.cast_zero, chromaticNumber_le_iff_colorable]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : IsEmpty V
    ⊢ G.Colorable 0
  -/
  apply colorable_of_isEmpty
  /-
    🎉 no goals
  -/


theorem isEmpty_of_chromaticNumber_eq_zero (G : SimpleGraph V) [Finite V]
    (h : G.chromaticNumber = 0) : IsEmpty V := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Finite V
    h : Eq G.chromaticNumber 0
    ⊢ IsEmpty V
  -/
  have h' := G.colorable_chromaticNumber_of_fintype
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Finite V
    h : Eq G.chromaticNumber 0
    h' : G.Colorable G.chromaticNumber.toNat
    ⊢ IsEmpty V
  -/
  rw [h] at h'
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Finite V
    h : Eq G.chromaticNumber 0
    h' : G.Colorable (ENat.toNat 0)
    ⊢ IsEmpty V
  -/
  exact G.isEmpty_of_colorable_zero h'
  /-
    🎉 no goals
  -/


theorem chromaticNumber_pos [Nonempty V] {n : ℕ} (hc : G.Colorable n) : 0 < G.chromaticNumber := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    ⊢ LT.lt 0 G.chromaticNumber
  -/
  rw [hc.chromaticNumber_eq_sInf, Nat.cast_pos]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    ⊢ LT.lt 0 (InfSet.sInf (setOf fun n' => G.Colorable n'))
  -/
  apply le_csInf (colorable_set_nonempty_of_colorable hc)
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    ⊢ ∀ (b : Nat), Membership.mem (setOf fun n => G.Colorable n) b → LE.le (Nat.su …
  -/
  intro m hm
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    m : Nat
    hm : Membership.mem (setOf fun n => G.Colorable n) m
    ⊢ LE.le (Nat.succ 0) m
  -/
  by_contra h'
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    m : Nat
    hm : Membership.mem (setOf fun n => G.Colorable n) m
    h' : Not (LE.le (Nat.succ 0) m)
    ⊢ False
  -/
  simp only [not_le] at h'
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    m : Nat
    hm : Membership.mem (setOf fun n => G.Colorable n) m
    h' : LT.lt m (Nat.succ 0)
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := hm.some (Classical.arbitrary V)
  /-
    case mk
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    m : Nat
    hm : Membership.mem (setOf fun n => G.Colorable n) m
    h' : LT.lt m (Nat.succ 0)
    i : Nat
    hi : LT.lt i m
    ⊢ False
  -/
  have h₁ : i < 0 := lt_of_lt_of_le hi (Nat.le_of_lt_succ h')
  /-
    case mk
    V : Type u
    G : SimpleGraph V
    inst✝ : Nonempty V
    n : Nat
    hc : G.Colorable n
    m : Nat
    hm : Membership.mem (setOf fun n => G.Colorable n) m
    h' : LT.lt m (Nat.succ 0)
    i : Nat
    hi : LT.lt i m
    h₁ : LT.lt i 0
    ⊢ False
  -/
  exact Nat.not_lt_zero _ h₁
  /-
    🎉 no goals
  -/


theorem colorable_of_chromaticNumber_ne_top (h : G.chromaticNumber ≠ ⊤) :
    G.Colorable (ENat.toNat G.chromaticNumber) := by
  /-
    V : Type u
    G : SimpleGraph V
    h : Ne G.chromaticNumber Top.top
    ⊢ G.Colorable G.chromaticNumber.toNat
  -/
  rw [chromaticNumber_ne_top_iff_exists] at h
  /-
    V : Type u
    G : SimpleGraph V
    h : Exists fun n => G.Colorable n
    ⊢ G.Colorable G.chromaticNumber.toNat
  -/
  obtain ⟨n, hn⟩ := h
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hn : G.Colorable n
    ⊢ G.Colorable G.chromaticNumber.toNat
  -/
  exact colorable_chromaticNumber hn
  /-
    🎉 no goals
  -/


theorem Colorable.mono_left {G' : SimpleGraph V} (h : G ≤ G') {n : ℕ} (hc : G'.Colorable n) :
    G.Colorable n :=
  ⟨hc.some.comp (Hom.mapSpanningSubgraphs h)⟩


theorem chromaticNumber_le_of_forall_imp {V' : Type*} {G' : SimpleGraph V'}
    (h : ∀ n, G'.Colorable n → G.Colorable n) :
    G.chromaticNumber ≤ G'.chromaticNumber := by
  /-
    V : Type u
    G : SimpleGraph V
    V' : Type u_3
    G' : SimpleGraph V'
    h : ∀ (n : Nat), G'.Colorable n → G.Colorable n
    ⊢ LE.le G.chromaticNumber G'.chromaticNumber
  -/
  rw [chromaticNumber, chromaticNumber]
  /-
    V : Type u
    G : SimpleGraph V
    V' : Type u_3
    G' : SimpleGraph V'
    h : ∀ (n : Nat), G'.Colorable n → G.Colorable n
    ⊢ LE.le (iInf fun n => iInf fun h => ↑n) (iInf fun n => iInf fun h => ↑n)
  -/
  simp only [Set.mem_setOf_eq, le_iInf_iff]
  /-
    V : Type u
    G : SimpleGraph V
    V' : Type u_3
    G' : SimpleGraph V'
    h : ∀ (n : Nat), G'.Colorable n → G.Colorable n
    ⊢ ∀ (i : Nat), G'.Colorable i → LE.le (iInf fun n => iInf fun h => ↑n) ↑i
  -/
  intro m hc
  /-
    V : Type u
    G : SimpleGraph V
    V' : Type u_3
    G' : SimpleGraph V'
    h : ∀ (n : Nat), G'.Colorable n → G.Colorable n
    m : Nat
    hc : G'.Colorable m
    ⊢ LE.le (iInf fun n => iInf fun h => ↑n) ↑m
  -/
  have := h _ hc
  /-
    V : Type u
    G : SimpleGraph V
    V' : Type u_3
    G' : SimpleGraph V'
    h : ∀ (n : Nat), G'.Colorable n → G.Colorable n
    m : Nat
    hc : G'.Colorable m
    this : G.Colorable m
    ⊢ LE.le (iInf fun n => iInf fun h => ↑n) ↑m
  -/
  rw [← chromaticNumber_le_iff_colorable] at this
  /-
    V : Type u
    G : SimpleGraph V
    V' : Type u_3
    G' : SimpleGraph V'
    h : ∀ (n : Nat), G'.Colorable n → G.Colorable n
    m : Nat
    hc : G'.Colorable m
    this : LE.le G.chromaticNumber ↑m
    ⊢ LE.le (iInf fun n => iInf fun h => ↑n) ↑m
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem chromaticNumber_mono (G' : SimpleGraph V)
    (h : G ≤ G') : G.chromaticNumber ≤ G'.chromaticNumber :=
  chromaticNumber_le_of_forall_imp fun _ => Colorable.mono_left h


theorem chromaticNumber_mono_of_embedding {V' : Type*} {G' : SimpleGraph V'}
    (f : G ↪g G') : G.chromaticNumber ≤ G'.chromaticNumber :=
  chromaticNumber_le_of_forall_imp fun _ => Colorable.of_embedding f


lemma card_le_chromaticNumber_iff_forall_surjective [Fintype α] :
    card α ≤ G.chromaticNumber ↔ ∀ C : G.Coloring α, Surjective C := by
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    inst✝ : Fintype α
    ⊢ Iff (LE.le (↑(Fintype.card α)) G.chromaticNumber) (∀ (C : G.Coloring α), Fun …
  -/
  refine ⟨fun h C ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : LE.le (↑(Fintype.card α)) G.chromaticNumber
      C : G.Coloring α
      ⊢ Function.Surjective ⇑C
    -/
  · rw [C.colorable.chromaticNumber_eq_sInf, Nat.cast_le] at h
    /-
      case refine_1
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : LE.le (Fintype.card α) (InfSet.sInf (setOf fun n' => G.Colorable n'))
      C : G.Coloring α
      ⊢ Function.Surjective ⇑C
    -/
    intro i
    /-
      case refine_1
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : LE.le (Fintype.card α) (InfSet.sInf (setOf fun n' => G.Colorable n'))
      C : G.Coloring α
      i : α
      ⊢ Exists fun a => Eq (C a) i
    -/
    by_contra! hi
    /-
      case refine_1
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : LE.le (Fintype.card α) (InfSet.sInf (setOf fun n' => G.Colorable n'))
      C : G.Coloring α
      i : α
      hi : ∀ (a : V), Ne (C a) i
      ⊢ False
    -/
    let D : G.Coloring {a // a ≠ i} := ⟨fun v ↦ ⟨C v, hi v⟩, (C.valid · <| congr_arg Subtype.val ·)⟩
    classical
    exact Nat.not_mem_of_lt_sInf ((Nat.sub_one_lt_of_lt <| card_pos_iff.2 ⟨i⟩).trans_le h)
      ⟨G.recolorOfEquiv (equivOfCardEq <| by simp [Nat.pred_eq_sub_one]) D⟩
    /-
      case refine_2
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : ∀ (C : G.Coloring α), Function.Surjective ⇑C
      ⊢ LE.le (↑(Fintype.card α)) G.chromaticNumber
    -/
  · simp only [chromaticNumber, Set.mem_setOf_eq, le_iInf_iff, Nat.cast_le, exists_prop]
    /-
      case refine_2
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : ∀ (C : G.Coloring α), Function.Surjective ⇑C
      ⊢ ∀ (i : Nat), G.Colorable i → LE.le (Fintype.card α) i
    -/
    rintro i ⟨C⟩
    /-
      case refine_2.intro
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      h : ∀ (C : G.Coloring α), Function.Surjective ⇑C
      i : Nat
      C : G.Coloring (Fin i)
      ⊢ LE.le (Fintype.card α) i
    -/
    contrapose! h
    /-
      case refine_2.intro
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      i : Nat
      C : G.Coloring (Fin i)
      h : LT.lt i (Fintype.card α)
      ⊢ Exists fun C => Not (Function.Surjective ⇑C)
    -/
    refine ⟨G.recolorOfCardLE (by simpa using h.le) C, fun hC ↦ ?_⟩
    /-
      case refine_2.intro
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      i : Nat
      C : G.Coloring (Fin i)
      h : LT.lt i (Fintype.card α)
      hC : Function.Surjective ⇑((G.recolorOfCardLE ⋯) C)
      ⊢ False
    -/
    dsimp at hC
    /-
      case refine_2.intro
      V : Type u
      G : SimpleGraph V
      α : Type u_1
      inst✝ : Fintype α
      i : Nat
      C : G.Coloring (Fin i)
      h : LT.lt i (Fintype.card α)
      hC : Function.Surjective (Function.comp ⇑⋯.some ⇑C)
      ⊢ False
    -/
    simpa [h.not_le] using Fintype.card_le_of_surjective _ hC.of_comp
    /-
      🎉 no goals
    -/


lemma le_chromaticNumber_iff_forall_surjective :
    n ≤ G.chromaticNumber ↔ ∀ C : G.Coloring (Fin n), Surjective C := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    ⊢ Iff (LE.le (↑n) G.chromaticNumber) (∀ (C : G.Coloring (Fin n)), Function.Sur …
  -/
  simp [← card_le_chromaticNumber_iff_forall_surjective]
  /-
    🎉 no goals
  -/


lemma chromaticNumber_eq_card_iff_forall_surjective [Fintype α] (hG : G.Colorable (card α)) :
    G.chromaticNumber = card α ↔ ∀ C : G.Coloring α, Surjective C := by
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    inst✝ : Fintype α
    hG : G.Colorable (Fintype.card α)
    ⊢ Iff (Eq G.chromaticNumber ↑(Fintype.card α)) (∀ (C : G.Coloring α), Function …
  -/
  rw [← hG.chromaticNumber_le.ge_iff_eq, card_le_chromaticNumber_iff_forall_surjective]
  /-
    🎉 no goals
  -/


lemma chromaticNumber_eq_iff_forall_surjective (hG : G.Colorable n) :
    G.chromaticNumber = n ↔ ∀ C : G.Coloring (Fin n), Surjective C := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hG : G.Colorable n
    ⊢ Iff (Eq G.chromaticNumber ↑n) (∀ (C : G.Coloring (Fin n)), Function.Surjecti …
  -/
  rw [← hG.chromaticNumber_le.ge_iff_eq, le_chromaticNumber_iff_forall_surjective]
  /-
    🎉 no goals
  -/


theorem chromaticNumber_bot [Nonempty V] : (⊥ : SimpleGraph V).chromaticNumber = 1 := by
  /-
    V : Type u
    inst✝ : Nonempty V
    ⊢ Eq Bot.bot.chromaticNumber 1
  -/
  have : (⊥ : SimpleGraph V).Colorable 1 := ⟨.mk 0 <| by simp⟩
  /-
    V : Type u
    inst✝ : Nonempty V
    this : Bot.bot.Colorable 1
    ⊢ Eq Bot.bot.chromaticNumber 1
  -/
  exact this.chromaticNumber_le.antisymm <| Order.one_le_iff_pos.2 <| chromaticNumber_pos this
  /-
    🎉 no goals
  -/


@[simp]
theorem chromaticNumber_top [Fintype V] : (⊤ : SimpleGraph V).chromaticNumber = Fintype.card V := by
  /-
    V : Type u
    inst✝ : Fintype V
    ⊢ Eq Top.top.chromaticNumber ↑(Fintype.card V)
  -/
  rw [chromaticNumber_eq_card_iff_forall_surjective (selfColoring _).colorable]
  /-
    V : Type u
    inst✝ : Fintype V
    ⊢ ∀ (C : Top.top.Coloring V), Function.Surjective ⇑C
  -/
  intro C
  /-
    V : Type u
    inst✝ : Fintype V
    C : Top.top.Coloring V
    ⊢ Function.Surjective ⇑C
  -/
  rw [← Finite.injective_iff_surjective]
  /-
    V : Type u
    inst✝ : Fintype V
    C : Top.top.Coloring V
    ⊢ Function.Injective ⇑C
  -/
  intro v w
  /-
    V : Type u
    inst✝ : Fintype V
    C : Top.top.Coloring V
    v w : V
    ⊢ Eq (C v) (C w) → Eq v w
  -/
  contrapose
  /-
    V : Type u
    inst✝ : Fintype V
    C : Top.top.Coloring V
    v w : V
    ⊢ Not (Eq v w) → Not (Eq (C v) (C w))
  -/
  intro h
  /-
    V : Type u
    inst✝ : Fintype V
    C : Top.top.Coloring V
    v w : V
    h : Not (Eq v w)
    ⊢ Not (Eq (C v) (C w))
  -/
  exact C.valid h
  /-
    🎉 no goals
  -/


theorem chromaticNumber_top_eq_top_of_infinite (V : Type*) [Infinite V] :
    (⊤ : SimpleGraph V).chromaticNumber = ⊤ := by
  /-
    V : Type u_3
    inst✝ : Infinite V
    ⊢ Eq Top.top.chromaticNumber Top.top
  -/
  by_contra hc
  /-
    V : Type u_3
    inst✝ : Infinite V
    hc : Not (Eq Top.top.chromaticNumber Top.top)
    ⊢ False
  -/
  rw [← Ne, chromaticNumber_ne_top_iff_exists] at hc
  /-
    V : Type u_3
    inst✝ : Infinite V
    hc : Exists fun n => Top.top.Colorable n
    ⊢ False
  -/
  obtain ⟨n, ⟨hn⟩⟩ := hc
  /-
    case intro.intro
    V : Type u_3
    inst✝ : Infinite V
    n : Nat
    hn : Top.top.Coloring (Fin n)
    ⊢ False
  -/
  exact not_injective_infinite_finite _ hn.injective_of_top_hom
  /-
    🎉 no goals
  -/


/-- The bicoloring of a complete bipartite graph using whether a vertex
is on the left or on the right. -/
def CompleteBipartiteGraph.bicoloring (V W : Type*) : (completeBipartiteGraph V W).Coloring Bool :=
  Coloring.mk (fun v => v.isRight)
    (by
      /-
        V✝ : Type u
        G : SimpleGraph V✝
        n : Nat
        α : Type u_1
        β : Type u_2
        C : G.Coloring α
        V : Type u_3
        W : Type u_4
        ⊢ ∀ {v w : Sum V W}, (completeBipartiteGraph V W).Adj v w → Ne ((fun v => v.is …
      -/
      intro v w
      /-
        V✝ : Type u
        G : SimpleGraph V✝
        n : Nat
        α : Type u_1
        β : Type u_2
        C : G.Coloring α
        V : Type u_3
        W : Type u_4
        v w : Sum V W
        ⊢ (completeBipartiteGraph V W).Adj v w → Ne ((fun v => v.isRight) v) ((fun v = …
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
      cases v <;> cases w <;> simp)
                              /-
                                🎉 no goals
                              -/


theorem CompleteBipartiteGraph.chromaticNumber {V W : Type*} [Nonempty V] [Nonempty W] :
    (completeBipartiteGraph V W).chromaticNumber = 2 := by
  rw [← Nat.cast_two, chromaticNumber_eq_iff_forall_surjective
    (by simpa using (CompleteBipartiteGraph.bicoloring V W).colorable)]
  /-
    V : Type u_3
    W : Type u_4
    inst✝¹ : Nonempty V
    inst✝ : Nonempty W
    ⊢ ∀ (C : (completeBipartiteGraph V W).Coloring (Fin 2)), Function.Surjective ⇑C
  -/
  intro C b
  /-
    V : Type u_3
    W : Type u_4
    inst✝¹ : Nonempty V
    inst✝ : Nonempty W
    C : (completeBipartiteGraph V W).Coloring (Fin 2)
    b : Fin 2
    ⊢ Exists fun a => Eq (C a) b
  -/
  have v := Classical.arbitrary V
  /-
    V : Type u_3
    W : Type u_4
    inst✝¹ : Nonempty V
    inst✝ : Nonempty W
    C : (completeBipartiteGraph V W).Coloring (Fin 2)
    b : Fin 2
    v : V
    ⊢ Exists fun a => Eq (C a) b
  -/
  have w := Classical.arbitrary W
  /-
    V : Type u_3
    W : Type u_4
    inst✝¹ : Nonempty V
    inst✝ : Nonempty W
    C : (completeBipartiteGraph V W).Coloring (Fin 2)
    b : Fin 2
    v : V
    w : W
    ⊢ Exists fun a => Eq (C a) b
  -/
  have h : (completeBipartiteGraph V W).Adj (Sum.inl v) (Sum.inr w) := by simp
  /-
    V : Type u_3
    W : Type u_4
    inst✝¹ : Nonempty V
    inst✝ : Nonempty W
    C : (completeBipartiteGraph V W).Coloring (Fin 2)
    b : Fin 2
    v : V
    w : W
    h : (completeBipartiteGraph V W).Adj (Sum.inl v) (Sum.inr w)
    ⊢ Exists fun a => Eq (C a) b
  -/
  by_cases he : C (Sum.inl v) = b
    /-
      case pos
      V : Type u_3
      W : Type u_4
      inst✝¹ : Nonempty V
      inst✝ : Nonempty W
      C : (completeBipartiteGraph V W).Coloring (Fin 2)
      b : Fin 2
      v : V
      w : W
      h : (completeBipartiteGraph V W).Adj (Sum.inl v) (Sum.inr w)
      he : Eq (C (Sum.inl v)) b
      ⊢ Exists fun a => Eq (C a) b
    -/
  · exact ⟨_, he⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    V : Type u_3
    W : Type u_4
    inst✝¹ : Nonempty V
    inst✝ : Nonempty W
    C : (completeBipartiteGraph V W).Coloring (Fin 2)
    b : Fin 2
    v : V
    w : W
    h : (completeBipartiteGraph V W).Adj (Sum.inl v) (Sum.inr w)
    he : Not (Eq (C (Sum.inl v)) b)
    ⊢ Exists fun a => Eq (C a) b
  -/
  by_cases he' : C (Sum.inr w) = b
    /-
      case pos
      V : Type u_3
      W : Type u_4
      inst✝¹ : Nonempty V
      inst✝ : Nonempty W
      C : (completeBipartiteGraph V W).Coloring (Fin 2)
      b : Fin 2
      v : V
      w : W
      h : (completeBipartiteGraph V W).Adj (Sum.inl v) (Sum.inr w)
      he : Not (Eq (C (Sum.inl v)) b)
      he' : Eq (C (Sum.inr w)) b
      ⊢ Exists fun a => Eq (C a) b
    -/
  · exact ⟨_, he'⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      V : Type u_3
      W : Type u_4
      inst✝¹ : Nonempty V
      inst✝ : Nonempty W
      C : (completeBipartiteGraph V W).Coloring (Fin 2)
      b : Fin 2
      v : V
      w : W
      h : (completeBipartiteGraph V W).Adj (Sum.inl v) (Sum.inr w)
      he : Not (Eq (C (Sum.inl v)) b)
      he' : Not (Eq (C (Sum.inr w)) b)
      ⊢ Exists fun a => Eq (C a) b
    -/
  · simpa using two_lt_card_iff.2 ⟨_, _, _, C.valid h, he, he'⟩
    /-
      🎉 no goals
    -/


theorem IsClique.card_le_of_coloring {s : Finset V} (h : G.IsClique s) [Fintype α]
    (C : G.Coloring α) : s.card ≤ Fintype.card α := by
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    s : Finset V
    h : G.IsClique ↑s
    inst✝ : Fintype α
    C : G.Coloring α
    ⊢ LE.le s.card (Fintype.card α)
  -/
  rw [isClique_iff_induce_eq] at h
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    s : Finset V
    h : Eq (SimpleGraph.induce (↑s) G) Top.top
    inst✝ : Fintype α
    C : G.Coloring α
    ⊢ LE.le s.card (Fintype.card α)
  -/
  have f : G.induce ↑s ↪g G := Embedding.comap (Function.Embedding.subtype fun x => x ∈ ↑s) G
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    s : Finset V
    h : Eq (SimpleGraph.induce (↑s) G) Top.top
    inst✝ : Fintype α
    C : G.Coloring α
    f : (SimpleGraph.induce (↑s) G).Embedding G
    ⊢ LE.le s.card (Fintype.card α)
  -/
  rw [h] at f
  /-
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    s : Finset V
    h : Eq (SimpleGraph.induce (↑s) G) Top.top
    inst✝ : Fintype α
    C : G.Coloring α
    f : Top.top.Embedding G
    ⊢ LE.le s.card (Fintype.card α)
  -/
  convert Fintype.card_le_of_injective _ (C.comp f.toHom).injective_of_top_hom using 1
  /-
    case h.e'_3
    V : Type u
    G : SimpleGraph V
    α : Type u_1
    s : Finset V
    h : Eq (SimpleGraph.induce (↑s) G) Top.top
    inst✝ : Fintype α
    C : G.Coloring α
    f : Top.top.Embedding G
    ⊢ Eq s.card (Fintype.card ↑↑s)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsClique.card_le_of_colorable {s : Finset V} (h : G.IsClique s) {n : ℕ}
    (hc : G.Colorable n) : s.card ≤ n := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Finset V
    h : G.IsClique ↑s
    n : Nat
    hc : G.Colorable n
    ⊢ LE.le s.card n
  -/
  convert h.card_le_of_coloring hc.some
  /-
    case h.e'_4
    V : Type u
    G : SimpleGraph V
    s : Finset V
    h : G.IsClique ↑s
    n : Nat
    hc : G.Colorable n
    ⊢ Eq n (Fintype.card (Fin n))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsClique.card_le_chromaticNumber {s : Finset V} (h : G.IsClique s) :
    s.card ≤ G.chromaticNumber := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Finset V
    h : G.IsClique ↑s
    ⊢ LE.le (↑s.card) G.chromaticNumber
  -/
  obtain (hc | hc) := eq_or_ne G.chromaticNumber ⊤
    /-
      case inl
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Eq G.chromaticNumber Top.top
      ⊢ LE.le (↑s.card) G.chromaticNumber
    -/
  · rw [hc]
    /-
      case inl
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Eq G.chromaticNumber Top.top
      ⊢ LE.le (↑s.card) Top.top
    -/
    exact le_top
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Ne G.chromaticNumber Top.top
      ⊢ LE.le (↑s.card) G.chromaticNumber
    -/
  · have hc' := hc
    /-
      case inr
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc hc' : Ne G.chromaticNumber Top.top
      ⊢ LE.le (↑s.card) G.chromaticNumber
    -/
    rw [chromaticNumber_ne_top_iff_exists] at hc'
    /-
      case inr
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Ne G.chromaticNumber Top.top
      hc' : Exists fun n => G.Colorable n
      ⊢ LE.le (↑s.card) G.chromaticNumber
    -/
    obtain ⟨n, c⟩ := hc'
    /-
      case inr.intro
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Ne G.chromaticNumber Top.top
      n : Nat
      c : G.Colorable n
      ⊢ LE.le (↑s.card) G.chromaticNumber
    -/
    rw [← ENat.coe_toNat_eq_self] at hc
    /-
      case inr.intro
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Eq (↑G.chromaticNumber.toNat) G.chromaticNumber
      n : Nat
      c : G.Colorable n
      ⊢ LE.le (↑s.card) G.chromaticNumber
    -/
    rw [← hc, Nat.cast_le]
    /-
      case inr.intro
      V : Type u
      G : SimpleGraph V
      s : Finset V
      h : G.IsClique ↑s
      hc : Eq (↑G.chromaticNumber.toNat) G.chromaticNumber
      n : Nat
      c : G.Colorable n
      ⊢ LE.le s.card G.chromaticNumber.toNat
    -/
    exact h.card_le_of_colorable (colorable_chromaticNumber c)
    /-
      🎉 no goals
    -/


protected theorem Colorable.cliqueFree {n m : ℕ} (hc : G.Colorable n) (hm : n < m) :
    G.CliqueFree m := by
  /-
    V : Type u
    G : SimpleGraph V
    n m : Nat
    hc : G.Colorable n
    hm : LT.lt n m
    ⊢ G.CliqueFree m
  -/
  by_contra h
  /-
    V : Type u
    G : SimpleGraph V
    n m : Nat
    hc : G.Colorable n
    hm : LT.lt n m
    h : Not (G.CliqueFree m)
    ⊢ False
  -/
  simp only [CliqueFree, isNClique_iff, not_forall, Classical.not_not] at h
  /-
    V : Type u
    G : SimpleGraph V
    n m : Nat
    hc : G.Colorable n
    hm : LT.lt n m
    h : Exists fun x => And (G.IsClique ↑x) (Eq x.card m)
    ⊢ False
  -/
  obtain ⟨s, h, rfl⟩ := h
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : G.Colorable n
    s : Finset V
    h : G.IsClique ↑s
    hm : LT.lt n s.card
    ⊢ False
  -/
  exact Nat.lt_le_asymm hm (h.card_le_of_colorable hc)
  /-
    🎉 no goals
  -/


theorem cliqueFree_of_chromaticNumber_lt {n : ℕ} (hc : G.chromaticNumber < n) :
    G.CliqueFree n := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt G.chromaticNumber ↑n
    ⊢ G.CliqueFree n
  -/
  have hne : G.chromaticNumber ≠ ⊤ := hc.ne_top
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt G.chromaticNumber ↑n
    hne : Ne G.chromaticNumber Top.top
    ⊢ G.CliqueFree n
  -/
  obtain ⟨m, hc'⟩ := chromaticNumber_ne_top_iff_exists.mp hne
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt G.chromaticNumber ↑n
    hne : Ne G.chromaticNumber Top.top
    m : Nat
    hc' : G.Colorable m
    ⊢ G.CliqueFree n
  -/
  have := colorable_chromaticNumber hc'
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt G.chromaticNumber ↑n
    hne : Ne G.chromaticNumber Top.top
    m : Nat
    hc' : G.Colorable m
    this : G.Colorable G.chromaticNumber.toNat
    ⊢ G.CliqueFree n
  -/
  refine this.cliqueFree ?_
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt G.chromaticNumber ↑n
    hne : Ne G.chromaticNumber Top.top
    m : Nat
    hc' : G.Colorable m
    this : G.Colorable G.chromaticNumber.toNat
    ⊢ LT.lt G.chromaticNumber.toNat n
  -/
  rw [← ENat.coe_toNat_eq_self] at hne
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt G.chromaticNumber ↑n
    hne : Eq (↑G.chromaticNumber.toNat) G.chromaticNumber
    m : Nat
    hc' : G.Colorable m
    this : G.Colorable G.chromaticNumber.toNat
    ⊢ LT.lt G.chromaticNumber.toNat n
  -/
  rw [← hne] at hc
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    n : Nat
    hc : LT.lt ↑G.chromaticNumber.toNat ↑n
    hne : Eq (↑G.chromaticNumber.toNat) G.chromaticNumber
    m : Nat
    hc' : G.Colorable m
    this : G.Colorable G.chromaticNumber.toNat
    ⊢ LT.lt G.chromaticNumber.toNat n
  -/
  simpa using hc
  /-
    🎉 no goals
  -/


