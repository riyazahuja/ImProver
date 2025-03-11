/-- The type of relations for the language of graphs, consisting of a single binary relation `adj`.
-/
inductive graphRel : ℕ → Type
  | adj : graphRel 2
  deriving DecidableEq


/-- The language consisting of a single relation representing adjacency. -/
protected def graph : Language := ⟨fun _ => Empty, graphRel⟩
  deriving IsRelational


/-- The symbol representing the adjacency relation. -/
abbrev adj : Language.graph.Relations 2 := .adj


/-- Any simple graph can be thought of as a structure in the language of graphs. -/
def _root_.SimpleGraph.structure (G : SimpleGraph V) : Language.graph.Structure V where
  RelMap | .adj => (fun x => G.Adj (x 0) (x 1))


instance instSubsingleton : Subsingleton (Language.graph.Relations n) :=
      /-
        V : Type u
        n : Nat
        ⊢ ∀ (a b : FirstOrder.Language.graph.Relations n), Eq a b
      -/
  ⟨by rintro ⟨⟩ ⟨⟩; rfl⟩
                    /-
                      🎉 no goals
                    -/


/-- The theory of simple graphs. -/
protected def Theory.simpleGraph : Language.graph.Theory :=
  {adj.irreflexive, adj.symmetric}


@[simp]
theorem Theory.simpleGraph_model_iff [Language.graph.Structure V] :
    V ⊨ Theory.simpleGraph ↔
      (Irreflexive fun x y : V => RelMap adj ![x, y]) ∧
        Symmetric fun x y : V => RelMap adj ![x, y] := by
  /-
    V : Type u
    inst✝ : FirstOrder.Language.graph.Structure V
    ⊢ Iff (FirstOrder.Language.Theory.Model V FirstOrder.Language.Theory.simpleGra …
  -/
  simp [Theory.simpleGraph]
  /-
    🎉 no goals
  -/


instance simpleGraph_model (G : SimpleGraph V) :
    @Theory.Model _ V G.structure Theory.simpleGraph := by
  /-
    V : Type u
    n : Nat
    G : SimpleGraph V
    ⊢ FirstOrder.Language.Theory.Model V FirstOrder.Language.Theory.simpleGraph
  -/
  letI := G.structure
  /-
    V : Type u
    n : Nat
    G : SimpleGraph V
    this : FirstOrder.Language.graph.Structure V := G.structure
    ⊢ FirstOrder.Language.Theory.Model V FirstOrder.Language.Theory.simpleGraph
  -/
  rw [Theory.simpleGraph_model_iff]
  /-
    V : Type u
    n : Nat
    G : SimpleGraph V
    this : FirstOrder.Language.graph.Structure V := G.structure
    ⊢ And (Irreflexive fun x y => FirstOrder.Language.Structure.RelMap FirstOrder. …
  -/
  exact ⟨G.loopless, G.symm⟩
  /-
    🎉 no goals
  -/


/-- Any model of the theory of simple graphs represents a simple graph. -/
@[simps]
def simpleGraphOfStructure [Language.graph.Structure V] [V ⊨ Theory.simpleGraph] :
    SimpleGraph V where
  Adj x y := RelMap adj ![x, y]
  symm :=
    Relations.realize_symmetric.1
      (Theory.realize_sentence_of_mem Theory.simpleGraph
        (Set.mem_insert_of_mem _ (Set.mem_singleton _)))
  loopless :=
    Relations.realize_irreflexive.1
      (Theory.realize_sentence_of_mem Theory.simpleGraph (Set.mem_insert _ _))


@[simp]
theorem _root_.SimpleGraph.simpleGraphOfStructure (G : SimpleGraph V) :
    @simpleGraphOfStructure V G.structure _ = G := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Eq (FirstOrder.Language.simpleGraphOfStructure V) G
  -/
  ext
  /-
    case Adj.h.h.a
    V : Type u
    G : SimpleGraph V
    x✝¹ x✝ : V
    ⊢ Iff ((FirstOrder.Language.simpleGraphOfStructure V).Adj x✝¹ x✝) (G.Adj x✝¹ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem structure_simpleGraphOfStructure [S : Language.graph.Structure V] [V ⊨ Theory.simpleGraph] :
    (simpleGraphOfStructure V).structure = S := by
  /-
    V : Type u
    S : FirstOrder.Language.graph.Structure V
    inst✝ : FirstOrder.Language.Theory.Model V FirstOrder.Language.Theory.simpleGr …
    ⊢ Eq (FirstOrder.Language.simpleGraphOfStructure V).structure S
  -/
  ext
  case funMap n f xs =>
    exact isEmptyElim f
  case RelMap n r xs =>
    match n, r with
    | 2, .adj =>
      rw [iff_eq_eq]
      change RelMap adj ![xs 0, xs 1] = _
      refine congr rfl (funext ?_)
      simp [Fin.forall_fin_two]


theorem Theory.simpleGraph_isSatisfiable : Theory.IsSatisfiable Theory.simpleGraph :=
  ⟨@Theory.ModelType.of _ _ Unit (SimpleGraph.structure ⊥) _ _⟩


