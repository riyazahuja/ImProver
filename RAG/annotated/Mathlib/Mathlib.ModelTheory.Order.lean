/-- The type of relations for the language of orders, consisting of a single binary relation `le`.
-/
inductive orderRel : ℕ → Type
  | le : orderRel 2
  deriving DecidableEq


/-- The relational language consisting of a single relation representing `≤`. -/
protected def order : Language := ⟨fun _ => Empty, orderRel⟩
  deriving IsRelational


@[simp]
lemma forall_relations {P : ∀ (n) (_ : Language.order.Relations n), Prop} :
    (∀ {n} (R), P n R) ↔ P 2 .le := ⟨fun h => h _, fun h n R =>
      match n, R with
      | 2, .le => h⟩


instance instSubsingleton : Subsingleton (Language.order.Relations n) :=
      /-
        L : FirstOrder.Language
        α : Type w
        M : Type w'
        n : Nat
        ⊢ ∀ (a b : FirstOrder.Language.order.Relations n), Eq a b
      -/
  ⟨by rintro ⟨⟩ ⟨⟩; rfl⟩
                    /-
                      🎉 no goals
                    -/


                                                                /-
                                                                  L : FirstOrder.Language
                                                                  α : Type w
                                                                  M : Type w'
                                                                  n : Nat
                                                                  x : FirstOrder.Language.order.Relations 0
                                                                  ⊢ False
                                                                -/
instance : IsEmpty (Language.order.Relations 0) := ⟨fun x => by cases x⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


instance : Unique (Σ n, Language.order.Relations n) :=
  ⟨⟨⟨2, .le⟩⟩, fun ⟨n, R⟩ =>
      match n, R with
      | 2, .le => rfl⟩


instance : Unique Language.order.Symbols := ⟨⟨Sum.inr default⟩, by
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    ⊢ ∀ (a : FirstOrder.Language.order.Symbols), Eq a Inhabited.default
  -/
  have : IsEmpty (Σ n, Language.order.Functions n) := isEmpty_sigma.2 inferInstance
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    this : IsEmpty (Sigma fun n => FirstOrder.Language.order.Functions n)
    ⊢ ∀ (a : FirstOrder.Language.order.Symbols), Eq a Inhabited.default
  -/
  simp only [Symbols, Sum.forall, reduceCtorEq, Sum.inr.injEq, IsEmpty.forall_iff, true_and]
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    this : IsEmpty (Sigma fun n => FirstOrder.Language.order.Functions n)
    ⊢ ∀ (b : Sigma fun l => FirstOrder.Language.order.Relations l), Eq b Inhabited …
  -/
  exact Unique.eq_default⟩
  /-
    🎉 no goals
  -/


@[simp]
                                                  /-
                                                    ⊢ Eq FirstOrder.Language.order.card 1
                                                  -/
lemma card_eq_one : Language.order.card = 1 := by simp [card]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- A language is ordered if it has a symbol representing `≤`. -/
class IsOrdered (L : Language.{u, v}) where
  /-- The relation symbol representing `≤`. -/
  leSymb : L.Relations 2


instance : IsOrdered Language.order :=
  ⟨.le⟩


lemma order.relation_eq_leSymb : (R : Language.order.Relations 2) → R = leSymb
  | .le => rfl


/-- Joins two terms `t₁, t₂` in a formula representing `t₁ ≤ t₂`. -/
def Term.le (t₁ t₂ : L.Term (α ⊕ (Fin n))) : L.BoundedFormula α n :=
  leSymb.boundedFormula₂ t₁ t₂


/-- Joins two terms `t₁, t₂` in a formula representing `t₁ < t₂`. -/
def Term.lt (t₁ t₂ : L.Term (α ⊕ (Fin n))) : L.BoundedFormula α n :=
  t₁.le t₂ ⊓ ∼(t₂.le t₁)


/-- The language homomorphism sending the unique symbol `≤` of `Language.order` to `≤` in an ordered
 language. -/
@[simps] def orderLHom : Language.order →ᴸ L where
  onRelation | _, .le => leSymb


@[simp]
theorem orderLHom_leSymb :
    (orderLHom L).onRelation leSymb = (leSymb : L.Relations 2) :=
  rfl


@[simp]
theorem orderLHom_order : orderLHom Language.order = LHom.id Language.order :=
  LHom.funext (Subsingleton.elim _ _) (Subsingleton.elim _ _)


/-- The theory of preorders. -/
def preorderTheory : L.Theory :=
  {leSymb.reflexive, leSymb.transitive}


instance : Theory.IsUniversal L.preorderTheory := ⟨by
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝ : L.IsOrdered
    ⊢ ∀ ⦃φ : L.Sentence⦄, Membership.mem L.preorderTheory φ → FirstOrder.Language. …
  -/
  simp only [preorderTheory, Set.mem_insert_iff, Set.mem_singleton_iff, forall_eq_or_imp, forall_eq]
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝ : L.IsOrdered
    ⊢ And (FirstOrder.Language.BoundedFormula.IsUniversal FirstOrder.Language.IsOr …
  -/
  exact ⟨leSymb.isUniversal_reflexive, leSymb.isUniversal_transitive⟩⟩
  /-
    🎉 no goals
  -/


/-- The theory of partial orders. -/
def partialOrderTheory : L.Theory :=
  insert leSymb.antisymmetric L.preorderTheory


instance : Theory.IsUniversal L.partialOrderTheory :=
  Theory.IsUniversal.insert leSymb.isUniversal_antisymmetric


/-- The theory of linear orders. -/
def linearOrderTheory : L.Theory :=
  insert leSymb.total L.partialOrderTheory


instance : Theory.IsUniversal L.linearOrderTheory :=
  Theory.IsUniversal.insert leSymb.isUniversal_total


/-- A sentence indicating that an order has no top element:
$\forall x, \exists y, \neg y \le x$.   -/
def noTopOrderSentence : L.Sentence :=
  ∀'∃'∼((&1).le &0)


/-- A sentence indicating that an order has no bottom element:
$\forall x, \exists y, \neg x \le y$. -/
def noBotOrderSentence : L.Sentence :=
  ∀'∃'∼((&0).le &1)


/-- A sentence indicating that an order is dense:
$\forall x, \forall y, x < y \to \exists z, x < z \wedge z < y$. -/
def denselyOrderedSentence : L.Sentence :=
  ∀'∀'((&0).lt &1 ⟹ ∃'((&0).lt &2 ⊓ (&2).lt &1))


/-- The theory of dense linear orders without endpoints. -/
def dlo : L.Theory :=
  L.linearOrderTheory ∪ {L.noTopOrderSentence, L.noBotOrderSentence, L.denselyOrderedSentence}


instance [h : M ⊨ L.dlo] : M ⊨ L.linearOrderTheory := h.mono Set.subset_union_left


instance [h : M ⊨ L.linearOrderTheory] : M ⊨ L.partialOrderTheory := h.mono (Set.subset_insert _ _)


instance [h : M ⊨ L.partialOrderTheory] : M ⊨ L.preorderTheory := h.mono (Set.subset_insert _ _)


instance sum.instIsOrdered : IsOrdered (L.sum Language.order) :=
  ⟨Sum.inr IsOrdered.leSymb⟩


/-- Any linearly-ordered type is naturally a structure in the language `Language.order`.
This is not an instance, because sometimes the `Language.order.Structure` is defined first. -/
def orderStructure [LE M] : Language.order.Structure M where
  RelMap | .le => (fun x => x 0 ≤ x 1)


/-- A structure is ordered if its language has a `≤` symbol whose interpretation is `≤`. -/
class OrderedStructure [L.IsOrdered] [LE M] [L.Structure M] : Prop where
  relMap_leSymb : ∀ (x : Fin 2 → M), RelMap (leSymb : L.Relations 2) x ↔ (x 0 ≤ x 1)


attribute [simp] relMap_leSymb


instance [Language.order.Structure M] [Language.order.OrderedStructure M]
    [(orderLHom L).IsExpansionOn M] : L.OrderedStructure M where
  relMap_leSymb x := by
    /-
      L : FirstOrder.Language
      α : Type w
      M : Type w'
      n : Nat
      inst✝⁵ : L.IsOrdered
      inst✝⁴ : L.Structure M
      inst✝³ : LE M
      inst✝² : FirstOrder.Language.order.Structure M
      inst✝¹ : FirstOrder.Language.order.OrderedStructure M
      inst✝ : L.orderLHom.IsExpansionOn M
      x : Fin 2 → M
      ⊢ Iff (FirstOrder.Language.Structure.RelMap FirstOrder.Language.IsOrdered.leSy …
    -/
    rw [← orderLHom_leSymb L, LHom.IsExpansionOn.map_onRelation, relMap_leSymb]
    /-
      🎉 no goals
    -/


instance [Language.order.Structure M] [Language.order.OrderedStructure M] :
    LHom.IsExpansionOn (orderLHom L) M where
                       /-
                         L : FirstOrder.Language
                         α : Type w
                         M : Type w'
                         n : Nat
                         inst✝⁵ : L.IsOrdered
                         inst✝⁴ : L.Structure M
                         inst✝³ : LE M
                         inst✝² : L.OrderedStructure M
                         inst✝¹ : FirstOrder.Language.order.Structure M
                         inst✝ : FirstOrder.Language.order.OrderedStructure M
                         ⊢ ∀ {n : Nat} (R : FirstOrder.Language.order.Relations n) (x : Fin n → M), Eq  …
                       -/
  map_onRelation := by simp [order.relation_eq_leSymb]
                       /-
                         🎉 no goals
                       -/


instance (S : L.Substructure M) : L.OrderedStructure S := ⟨fun x => relMap_leSymb (S.subtype ∘ x)⟩


@[simp]
theorem Term.realize_le {t₁ t₂ : L.Term (α ⊕ (Fin n))} {v : α → M}
    {xs : Fin n → M} :
    (t₁.le t₂).Realize v xs ↔ t₁.realize (Sum.elim v xs) ≤ t₂.realize (Sum.elim v xs) := by
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    t₁ t₂ : L.Term (Sum α (Fin n))
    v : α → M
    xs : Fin n → M
    ⊢ Iff ((t₁.le t₂).Realize v xs) (LE.le (FirstOrder.Language.Term.realize (Sum. …
  -/
  simp [Term.le]
  /-
    🎉 no goals
  -/


theorem realize_noTopOrder_iff : M ⊨ L.noTopOrderSentence ↔ NoTopOrder M := by
  simp only [noTopOrderSentence, Sentence.Realize, Formula.Realize, BoundedFormula.realize_all,
    BoundedFormula.realize_ex, BoundedFormula.realize_not, Term.realize, Term.realize_le,
    Sum.elim_inr]
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    ⊢ Iff (∀ (a : M), Exists fun a_1 => Not (LE.le (Fin.snoc (Fin.snoc Inhabited.d …
  -/
  refine ⟨fun h => ⟨fun a => h a⟩, ?_⟩
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    ⊢ NoTopOrder M → ∀ (a : M), Exists fun a_1 => Not (LE.le (Fin.snoc (Fin.snoc I …
  -/
  intro h a
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    h : NoTopOrder M
    a : M
    ⊢ Exists fun a_1 => Not (LE.le (Fin.snoc (Fin.snoc Inhabited.default a) a_1 1) …
  -/
  exact exists_not_le a
  /-
    🎉 no goals
  -/


theorem realize_noBotOrder_iff : M ⊨ L.noBotOrderSentence ↔ NoBotOrder M := by
  simp only [noBotOrderSentence, Sentence.Realize, Formula.Realize, BoundedFormula.realize_all,
    BoundedFormula.realize_ex, BoundedFormula.realize_not, Term.realize, Term.realize_le,
    Sum.elim_inr]
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    ⊢ Iff (∀ (a : M), Exists fun a_1 => Not (LE.le (Fin.snoc (Fin.snoc Inhabited.d …
  -/
  refine ⟨fun h => ⟨fun a => h a⟩, ?_⟩
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    ⊢ NoBotOrder M → ∀ (a : M), Exists fun a_1 => Not (LE.le (Fin.snoc (Fin.snoc I …
  -/
  intro h a
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LE M
    inst✝ : L.OrderedStructure M
    h : NoBotOrder M
    a : M
    ⊢ Exists fun a_1 => Not (LE.le (Fin.snoc (Fin.snoc Inhabited.default a) a_1 0) …
  -/
  exact exists_not_ge a
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_noTopOrder [h : NoTopOrder M] : M ⊨ L.noTopOrderSentence :=
  realize_noTopOrder_iff.2 h


@[simp]
theorem realize_noBotOrder [h : NoBotOrder M] : M ⊨ L.noBotOrderSentence :=
  realize_noBotOrder_iff.2 h


theorem noTopOrder_of_dlo [M ⊨ L.dlo] : NoTopOrder M :=
  realize_noTopOrder_iff.1 (L.dlo.realize_sentence_of_mem (by
    /-
      L : FirstOrder.Language
      M : Type w'
      inst✝⁴ : L.IsOrdered
      inst✝³ : L.Structure M
      inst✝² : LE M
      inst✝¹ : L.OrderedStructure M
      inst✝ : FirstOrder.Language.Theory.Model M L.dlo
      ⊢ Membership.mem L.dlo L.noTopOrderSentence
    -/
    simp only [dlo, Set.union_insert, Set.union_singleton, Set.mem_insert_iff, true_or]))
    /-
      🎉 no goals
    -/


theorem noBotOrder_of_dlo [M ⊨ L.dlo] : NoBotOrder M :=
  realize_noBotOrder_iff.1 (L.dlo.realize_sentence_of_mem (by
    /-
      L : FirstOrder.Language
      M : Type w'
      inst✝⁴ : L.IsOrdered
      inst✝³ : L.Structure M
      inst✝² : LE M
      inst✝¹ : L.OrderedStructure M
      inst✝ : FirstOrder.Language.Theory.Model M L.dlo
      ⊢ Membership.mem L.dlo L.noBotOrderSentence
    -/
    simp only [dlo, Set.union_insert, Set.union_singleton, Set.mem_insert_iff, true_or, or_true]))
    /-
      🎉 no goals
    -/


@[simp]
theorem orderedStructure_iff
    [LE M] [Language.order.Structure M] [Language.order.OrderedStructure M] :
    L.OrderedStructure M ↔ LHom.IsExpansionOn (orderLHom L) M :=
  ⟨fun _ => inferInstance, fun _ => inferInstance⟩


instance model_preorder : M ⊨ L.preorderTheory := by
  simp only [preorderTheory, Theory.model_insert_iff, Relations.realize_reflexive, relMap_leSymb,
    Theory.model_singleton_iff, Relations.realize_transitive]
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : Preorder M
    inst✝ : L.OrderedStructure M
    ⊢ And (Reflexive fun x y => LE.le (Matrix.vecCons x (Matrix.vecCons y Matrix.v …
  -/
  exact ⟨le_refl, fun _ _ _ => le_trans⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem Term.realize_lt {t₁ t₂ : L.Term (α ⊕ (Fin n))}
    {v : α → M} {xs : Fin n → M} :
    (t₁.lt t₂).Realize v xs ↔ t₁.realize (Sum.elim v xs) < t₂.realize (Sum.elim v xs) := by
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : Preorder M
    inst✝ : L.OrderedStructure M
    t₁ t₂ : L.Term (Sum α (Fin n))
    v : α → M
    xs : Fin n → M
    ⊢ Iff ((t₁.lt t₂).Realize v xs) (LT.lt (FirstOrder.Language.Term.realize (Sum. …
  -/
  simp [Term.lt, lt_iff_le_not_le]
  /-
    🎉 no goals
  -/


theorem realize_denselyOrdered_iff :
    M ⊨ L.denselyOrderedSentence ↔ DenselyOrdered M := by
  simp only [denselyOrderedSentence, Sentence.Realize, Formula.Realize,
    BoundedFormula.realize_imp, BoundedFormula.realize_all, Term.realize, Term.realize_lt,
    Sum.elim_inr, BoundedFormula.realize_ex, BoundedFormula.realize_inf]
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : Preorder M
    inst✝ : L.OrderedStructure M
    ⊢ Iff (∀ (a a_1 : M), LT.lt (Fin.snoc (Fin.snoc Inhabited.default a) a_1 0) (F …
  -/
  refine ⟨fun h => ⟨fun a b ab => h a b ab⟩, ?_⟩
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : Preorder M
    inst✝ : L.OrderedStructure M
    ⊢ DenselyOrdered M → ∀ (a a_1 : M), LT.lt (Fin.snoc (Fin.snoc Inhabited.defaul …
  -/
  intro h a b ab
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : Preorder M
    inst✝ : L.OrderedStructure M
    h : DenselyOrdered M
    a b : M
    ab : LT.lt (Fin.snoc (Fin.snoc Inhabited.default a) b 0) (Fin.snoc (Fin.snoc I …
    ⊢ Exists fun a_1 => And (LT.lt (Fin.snoc (Fin.snoc (Fin.snoc Inhabited.default …
  -/
  exact exists_between ab
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_denselyOrdered [h : DenselyOrdered M] :
    M ⊨ L.denselyOrderedSentence :=
  realize_denselyOrdered_iff.2 h


theorem denselyOrdered_of_dlo [M ⊨ L.dlo] : DenselyOrdered M :=
  realize_denselyOrdered_iff.1 (L.dlo.realize_sentence_of_mem (by
    /-
      L : FirstOrder.Language
      M : Type w'
      inst✝⁴ : L.IsOrdered
      inst✝³ : L.Structure M
      inst✝² : Preorder M
      inst✝¹ : L.OrderedStructure M
      inst✝ : FirstOrder.Language.Theory.Model M L.dlo
      ⊢ Membership.mem L.dlo L.denselyOrderedSentence
    -/
    simp only [dlo, Set.union_insert, Set.union_singleton, Set.mem_insert_iff, true_or, or_true]))
    /-
      🎉 no goals
    -/


instance model_partialOrder [PartialOrder M] [L.OrderedStructure M] :
    M ⊨ L.partialOrderTheory := by
  simp only [partialOrderTheory, Theory.model_insert_iff, Relations.realize_antisymmetric,
    relMap_leSymb, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    model_preorder, and_true]
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : PartialOrder M
    inst✝ : L.OrderedStructure M
    ⊢ AntiSymmetric fun x y => LE.le x y
  -/
  exact fun _ _ => le_antisymm
  /-
    🎉 no goals
  -/


instance model_linearOrder : M ⊨ L.linearOrderTheory := by
  simp only [linearOrderTheory, Theory.model_insert_iff, Relations.realize_total, relMap_leSymb,
    Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, model_partialOrder,
    and_true]
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝³ : L.IsOrdered
    inst✝² : L.Structure M
    inst✝¹ : LinearOrder M
    inst✝ : L.OrderedStructure M
    ⊢ Total fun x y => LE.le x y
  -/
  exact le_total
  /-
    🎉 no goals
  -/


instance model_dlo [DenselyOrdered M] [NoTopOrder M] [NoBotOrder M] :
    M ⊨ L.dlo := by
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝⁶ : L.IsOrdered
    inst✝⁵ : L.Structure M
    inst✝⁴ : LinearOrder M
    inst✝³ : L.OrderedStructure M
    inst✝² : DenselyOrdered M
    inst✝¹ : NoTopOrder M
    inst✝ : NoBotOrder M
    ⊢ FirstOrder.Language.Theory.Model M L.dlo
  -/
  simp [dlo, model_linearOrder, Theory.model_insert_iff]
  /-
    🎉 no goals
  -/


/-- Any structure in an ordered language can be ordered correspondingly. -/
def leOfStructure : LE M where
  le a b := Structure.RelMap (leSymb : L.Relations 2) ![a,b]


instance : @OrderedStructure L M _ (L.leOfStructure M) _ := by
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝¹ : L.IsOrdered
    inst✝ : L.Structure M
    ⊢ L.OrderedStructure M
  -/
  letI := L.leOfStructure M
  /-
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝¹ : L.IsOrdered
    inst✝ : L.Structure M
    this : LE M := L.leOfStructure M
    ⊢ L.OrderedStructure M
  -/
  constructor
  /-
    case relMap_leSymb
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝¹ : L.IsOrdered
    inst✝ : L.Structure M
    this : LE M := L.leOfStructure M
    ⊢ ∀ (x : Fin 2 → M), Iff (FirstOrder.Language.Structure.RelMap FirstOrder.Lang …
  -/
  simp only [Fin.forall_fin_succ_pi, Fin.cons_zero, Fin.forall_fin_zero_pi]
  /-
    case relMap_leSymb
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝¹ : L.IsOrdered
    inst✝ : L.Structure M
    this : LE M := L.leOfStructure M
    ⊢ ∀ (a a_1 : M), Iff (FirstOrder.Language.Structure.RelMap FirstOrder.Language …
  -/
  intros
  /-
    case relMap_leSymb
    L : FirstOrder.Language
    α : Type w
    M : Type w'
    n : Nat
    inst✝¹ : L.IsOrdered
    inst✝ : L.Structure M
    this : LE M := L.leOfStructure M
    a✝¹ a✝ : M
    ⊢ Iff (FirstOrder.Language.Structure.RelMap FirstOrder.Language.IsOrdered.leSy …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The order structure on an ordered language is decidable. -/
-- This should not be a global instance,
-- because it will match with any `LE` typeclass search
@[local instance]
def decidableLEOfStructure
    [h : DecidableRel (fun (a b : M) => Structure.RelMap (leSymb : L.Relations 2) ![a,b])] :
    letI := L.leOfStructure M
    DecidableRel ((· : M) ≤ ·) := h


/-- Any model of a theory of preorders is a preorder. -/
def preorderOfModels [h : M ⊨ L.preorderTheory] : Preorder M where
  __ := L.leOfStructure M
  le_refl := Relations.realize_reflexive.1 ((Theory.model_iff _).1 h _
        /-
          L : FirstOrder.Language
          α : Type w
          M : Type w'
          n : Nat
          inst✝¹ : L.IsOrdered
          inst✝ : L.Structure M
          h : FirstOrder.Language.Theory.Model M L.preorderTheory
          ⊢ Membership.mem L.preorderTheory FirstOrder.Language.IsOrdered.leSymb.reflexive
        -/
    (by simp only [preorderTheory, Set.mem_insert_iff, Set.mem_singleton_iff, true_or]))
        /-
          🎉 no goals
        -/
  le_trans := Relations.realize_transitive.1 ((Theory.model_iff _).1 h _
        /-
          L : FirstOrder.Language
          α : Type w
          M : Type w'
          n : Nat
          inst✝¹ : L.IsOrdered
          inst✝ : L.Structure M
          h : FirstOrder.Language.Theory.Model M L.preorderTheory
          ⊢ Membership.mem L.preorderTheory FirstOrder.Language.IsOrdered.leSymb.transit …
        -/
    (by simp only [preorderTheory, Set.mem_insert_iff, Set.mem_singleton_iff, or_true]))
        /-
          🎉 no goals
        -/


/-- Any model of a theory of partial orders is a partial order. -/
def partialOrderOfModels [h : M ⊨ L.partialOrderTheory] : PartialOrder M where
  __ := L.preorderOfModels M
  le_antisymm := Relations.realize_antisymmetric.1 ((Theory.model_iff _).1 h _
        /-
          L : FirstOrder.Language
          α : Type w
          M : Type w'
          n : Nat
          inst✝¹ : L.IsOrdered
          inst✝ : L.Structure M
          h : FirstOrder.Language.Theory.Model M L.partialOrderTheory
          ⊢ Membership.mem L.partialOrderTheory FirstOrder.Language.IsOrdered.leSymb.ant …
        -/
    (by simp only [partialOrderTheory, Set.mem_insert_iff, Set.mem_singleton_iff, true_or]))
        /-
          🎉 no goals
        -/


/-- Any model of a theory of linear orders is a linear order. -/
def linearOrderOfModels [h : M ⊨ L.linearOrderTheory]
    [DecidableRel (fun (a b : M) => Structure.RelMap (leSymb : L.Relations 2) ![a,b])] :
    LinearOrder M where
  __ := L.partialOrderOfModels M
  le_total := Relations.realize_total.1 ((Theory.model_iff _).1 h _
        /-
          L : FirstOrder.Language
          α : Type w
          M : Type w'
          n : Nat
          inst✝² : L.IsOrdered
          inst✝¹ : L.Structure M
          h : FirstOrder.Language.Theory.Model M L.linearOrderTheory
          inst✝ : DecidableRel fun a b => FirstOrder.Language.Structure.RelMap FirstOrde …
          ⊢ Membership.mem L.linearOrderTheory FirstOrder.Language.IsOrdered.leSymb.total
        -/
    (by simp only [linearOrderTheory, Set.mem_insert_iff, Set.mem_singleton_iff, true_or]))
        /-
          🎉 no goals
        -/
  decidableLE := inferInstance


instance [FunLike F M N] [OrderHomClass F M N] : Language.order.HomClass F M N :=
  ⟨fun _ => isEmptyElim, by
    simp only [forall_relations, relation_eq_leSymb, relMap_leSymb, Fin.isValue,
      Function.comp_apply]
    /-
      L : FirstOrder.Language
      α : Type w
      M : Type w'
      n : Nat
      inst✝⁷ : FirstOrder.Language.order.Structure M
      inst✝⁶ : LE M
      inst✝⁵ : FirstOrder.Language.order.OrderedStructure M
      N : Type u_1
      inst✝⁴ : FirstOrder.Language.order.Structure N
      inst✝³ : LE N
      inst✝² : FirstOrder.Language.order.OrderedStructure N
      F : Type u_2
      inst✝¹ : FunLike F M N
      inst✝ : OrderHomClass F M N
      ⊢ ∀ (φ : F) (x : Fin 2 → M), LE.le (x 0) (x 1) → LE.le (φ (x 0)) (φ (x 1))
    -/
    exact fun φ x => map_rel φ⟩
    /-
      🎉 no goals
    -/

-- If `OrderEmbeddingClass` or `RelEmbeddingClass` is defined, this should be generalized.

instance : Language.order.StrongHomClass (M ↪o N) M N :=
  ⟨fun _ => isEmptyElim,
    by simp only [order.forall_relations, order.relation_eq_leSymb, relMap_leSymb, Fin.isValue,
    Function.comp_apply, RelEmbedding.map_rel_iff, implies_true]⟩


instance [EquivLike F M N] [OrderIsoClass F M N] : Language.order.StrongHomClass F M N :=
  ⟨fun _ => isEmptyElim,
    by simp only [order.forall_relations, order.relation_eq_leSymb, relMap_leSymb, Fin.isValue,
      Function.comp_apply, map_le_map_iff, implies_true]⟩


lemma monotone [Preorder M] [L.OrderedStructure M] [Preorder N] [L.OrderedStructure N] (f : F) :
    Monotone f := fun a b => by
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝⁸ : L.IsOrdered
    inst✝⁷ : L.Structure M
    N : Type u_1
    inst✝⁶ : L.Structure N
    F : Type u_2
    inst✝⁵ : FunLike F M N
    inst✝⁴ : L.HomClass F M N
    inst✝³ : Preorder M
    inst✝² : L.OrderedStructure M
    inst✝¹ : Preorder N
    inst✝ : L.OrderedStructure N
    f : F
    a b : M
    ⊢ LE.le a b → LE.le (f a) (f b)
  -/
  have h := HomClass.map_rel f leSymb ![a,b]
  simp only [relMap_leSymb, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Function.comp_apply] at h
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝⁸ : L.IsOrdered
    inst✝⁷ : L.Structure M
    N : Type u_1
    inst✝⁶ : L.Structure N
    F : Type u_2
    inst✝⁵ : FunLike F M N
    inst✝⁴ : L.HomClass F M N
    inst✝³ : Preorder M
    inst✝² : L.OrderedStructure M
    inst✝¹ : Preorder N
    inst✝ : L.OrderedStructure N
    f : F
    a b : M
    h : LE.le a b → LE.le (f a) (f b)
    ⊢ LE.le a b → LE.le (f a) (f b)
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma strictMono [EmbeddingLike F M N] [PartialOrder M] [L.OrderedStructure M]
    [PartialOrder N] [L.OrderedStructure N] (f : F) :
    StrictMono f :=
  (HomClass.monotone f).strictMono_of_injective (EmbeddingLike.injective f)


/-- This is not an instance because it would form a loop with
  `FirstOrder.Language.order.instStrongHomClassOfOrderIsoClass`.
  As both types are `Prop`s, it would only cause a slowdown.  -/
lemma StrongHomClass.toOrderIsoClass
    (L : Language) [L.IsOrdered] (M : Type*) [L.Structure M] [LE M] [L.OrderedStructure M]
    (N : Type*) [L.Structure N] [LE N] [L.OrderedStructure N]
    (F : Type*) [EquivLike F M N] [L.StrongHomClass F M N] :
    OrderIsoClass F M N where
  map_le_map_iff f a b := by
    /-
      L : FirstOrder.Language
      inst✝⁸ : L.IsOrdered
      M : Type u_1
      inst✝⁷ : L.Structure M
      inst✝⁶ : LE M
      inst✝⁵ : L.OrderedStructure M
      N : Type u_2
      inst✝⁴ : L.Structure N
      inst✝³ : LE N
      inst✝² : L.OrderedStructure N
      F : Type u_3
      inst✝¹ : EquivLike F M N
      inst✝ : L.StrongHomClass F M N
      f : F
      a b : M
      ⊢ Iff (LE.le (f a) (f b)) (LE.le a b)
    -/
    have h := StrongHomClass.map_rel f leSymb ![a,b]
    simp only [relMap_leSymb, Fin.isValue, Function.comp_apply, Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.head_cons] at h
    /-
      L : FirstOrder.Language
      inst✝⁸ : L.IsOrdered
      M : Type u_1
      inst✝⁷ : L.Structure M
      inst✝⁶ : LE M
      inst✝⁵ : L.OrderedStructure M
      N : Type u_2
      inst✝⁴ : L.Structure N
      inst✝³ : LE N
      inst✝² : L.OrderedStructure N
      F : Type u_3
      inst✝¹ : EquivLike F M N
      inst✝ : L.StrongHomClass F M N
      f : F
      a b : M
      h : Iff (LE.le (f a) (f b)) (LE.le a b)
      ⊢ Iff (LE.le (f a) (f b)) (LE.le a b)
    -/
    exact h
    /-
      🎉 no goals
    -/


lemma dlo_isExtensionPair
    (M : Type w) [Language.order.Structure M] [M ⊨ Language.order.linearOrderTheory]
    (N : Type w') [Language.order.Structure N] [N ⊨ Language.order.dlo] [Nonempty N] :
    Language.order.IsExtensionPair M N := by
  classical
  rw [isExtensionPair_iff_exists_embedding_closure_singleton_sup]
  intro S S_fg f m
  letI := Language.order.linearOrderOfModels M
  letI := Language.order.linearOrderOfModels N
  have := Language.order.denselyOrdered_of_dlo N
  have := Language.order.noBotOrder_of_dlo N
  have := Language.order.noTopOrder_of_dlo N
  have := NoBotOrder.to_noMinOrder N
  have := NoTopOrder.to_noMaxOrder N
  have hS : Set.Finite (S : Set M) := (S.fg_iff_structure_fg.1 S_fg).finite
  obtain ⟨g, hg⟩ := Order.exists_orderEmbedding_insert hS.toFinset
    ((OrderIso.setCongr hS.toFinset (S : Set M) hS.coe_toFinset).toOrderEmbedding.trans
      (OrderEmbedding.ofStrictMono f (HomClass.strictMono f))) m
  let g' :
    ((Substructure.closure Language.order).toFun {m} ⊔ S : Language.order.Substructure M) ↪o N :=
    ((OrderIso.setCongr _ _ (by
      convert LowerAdjoint.closure_eq_self_of_mem_closed _
        (Substructure.mem_closed_of_isRelational Language.order
        ((insert m hS.toFinset : Finset M) : Set M))
      simp only [Finset.coe_insert, Set.Finite.coe_toFinset, Substructure.closure_insert,
        Substructure.closure_eq])).toOrderEmbedding.trans g)
  use StrongHomClass.toEmbedding g'
  ext ⟨x, xS⟩
  refine congr_fun hg.symm ⟨x, (?_ : x ∈ hS.toFinset)⟩
  simp only [Set.Finite.mem_toFinset, SetLike.mem_coe, xS]


instance (M : Type w) [Language.order.Structure M] [M ⊨ Language.order.dlo] [Nonempty M] :
    Infinite M := by
  /-
    L : FirstOrder.Language
    α : Type w
    M✝ : Type w'
    n : Nat
    M : Type w
    inst✝² : FirstOrder.Language.order.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M FirstOrder.Language.order.dlo
    inst✝ : Nonempty M
    ⊢ Infinite M
  -/
  letI := orderStructure ℚ
  /-
    L : FirstOrder.Language
    α : Type w
    M✝ : Type w'
    n : Nat
    M : Type w
    inst✝² : FirstOrder.Language.order.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M FirstOrder.Language.order.dlo
    inst✝ : Nonempty M
    this : FirstOrder.Language.order.Structure Rat := FirstOrder.Language.orderStr …
    ⊢ Infinite M
  -/
  obtain ⟨f, _⟩ := embedding_from_cg cg_of_countable default (dlo_isExtensionPair ℚ M)
  /-
    case intro
    L : FirstOrder.Language
    α : Type w
    M✝ : Type w'
    n : Nat
    M : Type w
    inst✝² : FirstOrder.Language.order.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M FirstOrder.Language.order.dlo
    inst✝ : Nonempty M
    this : FirstOrder.Language.order.Structure Rat := FirstOrder.Language.orderStr …
    f : FirstOrder.Language.order.Embedding Rat M
    h✝ : LE.le (↑Inhabited.default) f.toPartialEquiv
    ⊢ Infinite M
  -/
  exact Infinite.of_injective f f.injective
  /-
    🎉 no goals
  -/


lemma dlo_age [Language.order.Structure M] [Mdlo : M ⊨ Language.order.dlo] [Nonempty M] :
    Language.order.age M = {M : CategoryTheory.Bundled.{w'} Language.order.Structure |
      Finite M ∧ M ⊨ Language.order.linearOrderTheory} := by
  classical
  rw [age]
  ext N
  refine ⟨fun ⟨hF, h⟩ => ⟨hF.finite, Theory.IsUniversal.models_of_embedding h.some⟩,
    fun ⟨hF, h⟩ => ⟨FG.of_finite, ?_⟩⟩
  letI := Language.order.linearOrderOfModels M
  letI := Language.order.linearOrderOfModels N
  exact ⟨StrongHomClass.toEmbedding (nonempty_orderEmbedding_of_finite_infinite N M).some⟩


/-- Any countable nonempty model of the theory of dense linear orders is a Fraïssé limit of the
class of finite models of the theory of linear orders. -/
theorem isFraisseLimit_of_countable_nonempty_dlo (M : Type w)
    [Language.order.Structure M] [Countable M] [Nonempty M] [M ⊨ Language.order.dlo] :
    IsFraisseLimit {M : CategoryTheory.Bundled.{w} Language.order.Structure |
      Finite M ∧ M ⊨ Language.order.linearOrderTheory} M :=
  ⟨(isUltrahomogeneous_iff_IsExtensionPair cg_of_countable).2 (dlo_isExtensionPair M M), dlo_age M⟩


/-- The class of finite models of the theory of linear orders is Fraïssé. -/
theorem isFraisse_finite_linear_order :
    IsFraisse {M : CategoryTheory.Bundled.{0} Language.order.Structure |
      Finite M ∧ M ⊨ Language.order.linearOrderTheory} := by
  /-
    ⊢ FirstOrder.Language.IsFraisse (setOf fun M => And (Finite ↑M) (FirstOrder.La …
  -/
  letI : Language.order.Structure ℚ := orderStructure _
  /-
    this : FirstOrder.Language.order.Structure Rat := FirstOrder.Language.orderStr …
    ⊢ FirstOrder.Language.IsFraisse (setOf fun M => And (Finite ↑M) (FirstOrder.La …
  -/
  exact (isFraisseLimit_of_countable_nonempty_dlo ℚ).isFraisse
  /-
    🎉 no goals
  -/


/-- The theory of dense linear orders is `ℵ₀`-categorical. -/
theorem aleph0_categorical_dlo : (ℵ₀).Categorical Language.order.dlo := fun M₁ M₂ h₁ h₂ => by
  /-
    M₁ M₂ : FirstOrder.Language.order.dlo.ModelType
    h₁ : Eq (Cardinal.mk ↑M₁) Cardinal.aleph0
    h₂ : Eq (Cardinal.mk ↑M₂) Cardinal.aleph0
    ⊢ Nonempty (FirstOrder.Language.order.Equiv ↑M₁ ↑M₂)
  -/
  obtain ⟨_⟩ := denumerable_iff.2 h₁
  /-
    case intro
    M₁ M₂ : FirstOrder.Language.order.dlo.ModelType
    h₁ : Eq (Cardinal.mk ↑M₁) Cardinal.aleph0
    h₂ : Eq (Cardinal.mk ↑M₂) Cardinal.aleph0
    val✝ : Denumerable ↑M₁
    ⊢ Nonempty (FirstOrder.Language.order.Equiv ↑M₁ ↑M₂)
  -/
  obtain ⟨_⟩ := denumerable_iff.2 h₂
  exact (isFraisseLimit_of_countable_nonempty_dlo M₁).nonempty_equiv
    (isFraisseLimit_of_countable_nonempty_dlo M₂)


/-- The theory of dense linear orders is `ℵ₀`-complete. -/
theorem dlo_isComplete : Language.order.dlo.IsComplete :=
                                                        /-
                                                          ⊢ LE.le (Cardinal.lift.{0, 0} FirstOrder.Language.order.card) (Cardinal.lift.{ …
                                                        -/
  aleph0_categorical_dlo.{0}.isComplete ℵ₀ _ le_rfl (by simp [one_le_aleph0])
                                                        /-
                                                          🎉 no goals
                                                        -/
    ⟨by
      /-
        ⊢ FirstOrder.Language.order.dlo.ModelType
      -/
      letI : Language.order.Structure ℚ := orderStructure ℚ
      /-
        this : FirstOrder.Language.order.Structure Rat := FirstOrder.Language.orderStr …
        ⊢ FirstOrder.Language.order.dlo.ModelType
      -/
      exact Theory.ModelType.of _ ℚ⟩
      /-
        🎉 no goals
      -/
    fun _ => inferInstance


