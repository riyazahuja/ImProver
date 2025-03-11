/-- A language consisting of Skolem functions for another language.
Called `skolem₁` because it is the first step in building a Skolemization of a language. -/
@[simps]
def skolem₁ : Language :=
  ⟨fun n => L.BoundedFormula Empty (n + 1), fun _ => Empty⟩


theorem card_functions_sum_skolem₁ :
    #(Σ n, (L.sum L.skolem₁).Functions n) = #(Σ n, L.BoundedFormula Empty (n + 1)) := by
  /-
    L : FirstOrder.Language
    ⊢ Eq (Cardinal.mk (Sigma fun n => (L.sum L.skolem₁).Functions n)) (Cardinal.mk …
  -/
  simp only [card_functions_sum, skolem₁_Functions, mk_sigma, sum_add_distrib']
  /-
    L : FirstOrder.Language
    ⊢ Eq (HAdd.hAdd (Cardinal.sum fun i => Cardinal.lift.{max u v, u} (Cardinal.mk …
  -/
  conv_lhs => enter [2, 1, i]; rw [lift_id'.{u, v}]
  /-
    L : FirstOrder.Language
    ⊢ Eq (HAdd.hAdd (Cardinal.sum fun i => Cardinal.lift.{max u v, u} (Cardinal.mk …
  -/
  rw [add_comm, add_eq_max, max_eq_left]
    /-
      L : FirstOrder.Language
      ⊢ LE.le (Cardinal.sum fun i => Cardinal.lift.{max u v, u} (Cardinal.mk (L.Func …
    -/
  · refine sum_le_sum _ _ fun n => ?_
    /-
      L : FirstOrder.Language
      n : Nat
      ⊢ LE.le (Cardinal.lift.{max u v, u} (Cardinal.mk (L.Functions n))) (Cardinal.m …
    -/
    rw [← lift_le.{_, max u v}, lift_lift, lift_mk_le.{v}]
    /-
      L : FirstOrder.Language
      n : Nat
      ⊢ Nonempty (Function.Embedding (L.Functions n) (L.BoundedFormula Empty (HAdd.h …
    -/
    refine ⟨⟨fun f => (func f default).bdEqual (func f default), fun f g h => ?_⟩⟩
    /-
      L : FirstOrder.Language
      n : Nat
      f g : L.Functions n
      h : Eq ((fun f => (FirstOrder.Language.Term.func f Inhabited.default).bdEqual  …
      ⊢ Eq f g
    -/
    rcases h with ⟨rfl, ⟨rfl⟩⟩
    /-
      case refl
      L : FirstOrder.Language
      n : Nat
      f : L.Functions n
      ⊢ Eq f f
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      L : FirstOrder.Language
      ⊢ LE.le Cardinal.aleph0 (Cardinal.sum fun i => Cardinal.mk (L.BoundedFormula E …
    -/
  · rw [← mk_sigma]
    exact infinite_iff.1 (Infinite.of_injective (fun n => ⟨n, ⊥⟩) fun x y xy =>
      (Sigma.mk.inj_iff.1 xy).1)


theorem card_functions_sum_skolem₁_le : #(Σ n, (L.sum L.skolem₁).Functions n) ≤ max ℵ₀ L.card := by
  /-
    L : FirstOrder.Language
    ⊢ LE.le (Cardinal.mk (Sigma fun n => (L.sum L.skolem₁).Functions n)) (Max.max  …
  -/
  rw [card_functions_sum_skolem₁]
  /-
    L : FirstOrder.Language
    ⊢ LE.le (Cardinal.mk (Sigma fun n => L.BoundedFormula Empty (HAdd.hAdd n 1)))  …
  -/
  trans #(Σ n, L.BoundedFormula Empty n)
  · exact
      ⟨⟨Sigma.map Nat.succ fun _ => id,
          Nat.succ_injective.sigma_map fun _ => Function.injective_id⟩⟩
    /-
      L : FirstOrder.Language
      ⊢ LE.le (Cardinal.mk (Sigma fun n => L.BoundedFormula Empty n)) (Max.max Cardi …
    -/
  · refine _root_.trans BoundedFormula.card_le (lift_le.{max u v}.1 ?_)
    /-
      L : FirstOrder.Language
      ⊢ LE.le (Cardinal.lift.{max u v, max u v} (Max.max Cardinal.aleph0 (HAdd.hAdd  …
    -/
    simp only [mk_empty, lift_zero, lift_uzero, zero_add]
    /-
      L : FirstOrder.Language
      ⊢ LE.le (Cardinal.lift.{max u v, max u v} (Max.max Cardinal.aleph0 L.card)) (C …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The structure assigning each function symbol of `L.skolem₁` to a skolem function generated with
choice. -/
noncomputable instance skolem₁Structure : L.skolem₁.Structure M :=
  ⟨fun {_} φ x => Classical.epsilon fun a => φ.Realize default (Fin.snoc x a : _ → M), fun {_} r =>
    Empty.elim r⟩


theorem skolem₁_reduct_isElementary (S : (L.sum L.skolem₁).Substructure M) :
    (LHom.sumInl.substructureReduct S).IsElementary := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    S : (L.sum L.skolem₁).Substructure M
    ⊢ (FirstOrder.Language.LHom.sumInl.substructureReduct S).IsElementary
  -/
  apply (LHom.sumInl.substructureReduct S).isElementary_of_exists
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    S : (L.sum L.skolem₁).Substructure M
    ⊢ ∀ (n : Nat) (φ : L.BoundedFormula Empty (HAdd.hAdd n 1)) (x : Fin n → Subtyp …
  -/
  intro n φ x a h
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    S : (L.sum L.skolem₁).Substructure M
    n : Nat
    φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
    x : Fin n → Subtype fun x => Membership.mem (FirstOrder.Language.LHom.sumInl.s …
    a : M
    h : φ.Realize Inhabited.default (Fin.snoc (Function.comp Subtype.val x) a)
    ⊢ Exists fun b => φ.Realize Inhabited.default (Fin.snoc (Function.comp Subtype …
  -/
  let φ' : (L.sum L.skolem₁).Functions n := LHom.sumInr.onFunction φ
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    S : (L.sum L.skolem₁).Substructure M
    n : Nat
    φ : L.BoundedFormula Empty (HAdd.hAdd n 1)
    x : Fin n → Subtype fun x => Membership.mem (FirstOrder.Language.LHom.sumInl.s …
    a : M
    h : φ.Realize Inhabited.default (Fin.snoc (Function.comp Subtype.val x) a)
    φ' : (L.sum L.skolem₁).Functions n := FirstOrder.Language.LHom.sumInr.onFuncti …
    ⊢ Exists fun b => φ.Realize Inhabited.default (Fin.snoc (Function.comp Subtype …
  -/
  use ⟨funMap φ' ((↑) ∘ x), ?_⟩
  · exact Classical.epsilon_spec (p := fun a => BoundedFormula.Realize φ default
          (Fin.snoc (Subtype.val ∘ x) a)) ⟨a, h⟩
  · exact S.fun_mem (LHom.sumInr.onFunction φ) ((↑) ∘ x) (by
      exact fun i => (x i).2)


/-- Any `L.sum L.skolem₁`-substructure is an elementary `L`-substructure. -/
noncomputable def elementarySkolem₁Reduct (S : (L.sum L.skolem₁).Substructure M) :
    L.ElementarySubstructure M :=
  ⟨LHom.sumInl.substructureReduct S, S.skolem₁_reduct_isElementary⟩


theorem coeSort_elementarySkolem₁Reduct (S : (L.sum L.skolem₁).Substructure M) :
    (S.elementarySkolem₁Reduct : Type w) = S :=
  rfl


instance Substructure.elementarySkolem₁Reduct.instSmall :
    Small.{max u v} (⊥ : (L.sum L.skolem₁).Substructure M).elementarySkolem₁Reduct := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    ⊢ Small.{max u v, w} (Subtype fun x => Membership.mem Bot.bot.elementarySkolem …
  -/
  rw [coeSort_elementarySkolem₁Reduct]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    ⊢ Small.{max u v, w} (Subtype fun x => Membership.mem Bot.bot x)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem exists_small_elementarySubstructure : ∃ S : L.ElementarySubstructure M, Small.{max u v} S :=
  ⟨Substructure.elementarySkolem₁Reduct ⊥, inferInstance⟩


/-- The **Downward Löwenheim–Skolem theorem** :
  If `s` is a set in an `L`-structure `M` and `κ` an infinite cardinal such that
  `max (#s, L.card) ≤ κ` and `κ ≤ # M`, then `M` has an elementary substructure containing `s` of
  cardinality `κ`. -/
theorem exists_elementarySubstructure_card_eq (s : Set M) (κ : Cardinal.{w'}) (h1 : ℵ₀ ≤ κ)
    (h2 : Cardinal.lift.{w'} #s ≤ Cardinal.lift.{w} κ)
    (h3 : Cardinal.lift.{w'} L.card ≤ Cardinal.lift.{max u v} κ)
    (h4 : Cardinal.lift.{w} κ ≤ Cardinal.lift.{w'} #M) :
    ∃ S : L.ElementarySubstructure M, s ⊆ S ∧ Cardinal.lift.{w'} #S = Cardinal.lift.{w} κ := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h1 : LE.le Cardinal.aleph0 κ
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.lift.{w, w'} κ)
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    ⊢ Exists fun S => And (HasSubset.Subset s ↑S) (Eq (Cardinal.lift.{w', w} (Card …
  -/
  obtain ⟨s', hs'⟩ := Cardinal.le_mk_iff_exists_set.1 h4
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h1 : LE.le Cardinal.aleph0 κ
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.lift.{w, w'} κ)
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    s' : Set (ULift.{w', w} M)
    hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
    ⊢ Exists fun S => And (HasSubset.Subset s ↑S) (Eq (Cardinal.lift.{w', w} (Card …
  -/
  rw [← aleph0_le_lift.{_, w}] at h1
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h1 : LE.le Cardinal.aleph0 (Cardinal.lift.{w, w'} κ)
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.lift.{w, w'} κ)
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    s' : Set (ULift.{w', w} M)
    hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
    ⊢ Exists fun S => And (HasSubset.Subset s ↑S) (Eq (Cardinal.lift.{w', w} (Card …
  -/
  rw [← hs'] at h1 h2 ⊢
  refine
    ⟨elementarySkolem₁Reduct (closure (L.sum L.skolem₁) (s ∪ Equiv.ulift '' s')),
      (s.subset_union_left).trans subset_closure, ?_⟩
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    s' : Set (ULift.{w', w} M)
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
    h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
    hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
    ⊢ Eq (Cardinal.lift.{w', w} (Cardinal.mk (Subtype fun x => Membership.mem ((Fi …
  -/
  have h := mk_image_eq_lift _ s' Equiv.ulift.injective
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    s' : Set (ULift.{w', w} M)
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
    h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
    hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
    h : Eq (Cardinal.lift.{max w w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s' …
    ⊢ Eq (Cardinal.lift.{w', w} (Cardinal.mk (Subtype fun x => Membership.mem ((Fi …
  -/
  rw [lift_umax.{w, w'}, lift_id'.{w, w'}] at h
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    s' : Set (ULift.{w', w} M)
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
    h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
    hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
    h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
    ⊢ Eq (Cardinal.lift.{w', w} (Cardinal.mk (Subtype fun x => Membership.mem ((Fi …
  -/
  rw [coeSort_elementarySkolem₁Reduct, ← h, lift_inj]
  refine
    le_antisymm (lift_le.1 (lift_card_closure_le.trans ?_))
      (mk_le_mk_of_subset ((s.subset_union_right).trans subset_closure))
  /-
    case intro
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : Nonempty M
    inst✝ : L.Structure M
    s : Set M
    κ : Cardinal.{w'}
    h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
    h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
    s' : Set (ULift.{w', w} M)
    h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
    h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
    hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
    h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
    ⊢ LE.le (Max.max Cardinal.aleph0 (HAdd.hAdd (Cardinal.lift.{max u v, w} (Cardi …
  -/
  rw [max_le_iff, aleph0_le_lift, ← aleph0_le_lift.{_, w'}, h, add_eq_max, max_le_iff, lift_le]
    /-
      case intro
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : Nonempty M
      inst✝ : L.Structure M
      s : Set M
      κ : Cardinal.{w'}
      h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
      h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
      s' : Set (ULift.{w', w} M)
      h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
      h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
      hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
      h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
      ⊢ And (LE.le Cardinal.aleph0 (Cardinal.mk ↑s')) (And (LE.le (Cardinal.mk ↑(Uni …
    -/
  · refine ⟨h1, (mk_union_le _ _).trans ?_, (lift_le.2 card_functions_sum_skolem₁_le).trans ?_⟩
      /-
        case intro.refine_1
        L : FirstOrder.Language
        M : Type w
        inst✝¹ : Nonempty M
        inst✝ : L.Structure M
        s : Set M
        κ : Cardinal.{w'}
        h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
        h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
        s' : Set (ULift.{w', w} M)
        h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
        h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
        hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
        h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
        ⊢ LE.le (HAdd.hAdd (Cardinal.mk ↑s) (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s' …
      -/
    · rw [← lift_le, lift_add, h, add_comm, add_eq_max h1]
      /-
        case intro.refine_1
        L : FirstOrder.Language
        M : Type w
        inst✝¹ : Nonempty M
        inst✝ : L.Structure M
        s : Set M
        κ : Cardinal.{w'}
        h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
        h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
        s' : Set (ULift.{w', w} M)
        h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
        h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
        hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
        h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
        ⊢ LE.le (Max.max (Cardinal.mk ↑s') (Cardinal.lift.{w', w} (Cardinal.mk ↑s))) ( …
      -/
      exact max_le le_rfl h2
      /-
        🎉 no goals
      -/
    · rw [lift_max, lift_aleph0, max_le_iff, aleph0_le_lift, and_comm, ← lift_le.{w'},
        lift_lift, lift_lift, ← aleph0_le_lift, h]
      /-
        case intro.refine_2
        L : FirstOrder.Language
        M : Type w
        inst✝¹ : Nonempty M
        inst✝ : L.Structure M
        s : Set M
        κ : Cardinal.{w'}
        h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
        h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
        s' : Set (ULift.{w', w} M)
        h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
        h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
        hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
        h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
        ⊢ And (LE.le (Cardinal.lift.{max w w', max u v} L.card) (Cardinal.lift.{max (m …
      -/
      refine ⟨?_, h1⟩
      /-
        case intro.refine_2
        L : FirstOrder.Language
        M : Type w
        inst✝¹ : Nonempty M
        inst✝ : L.Structure M
        s : Set M
        κ : Cardinal.{w'}
        h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
        h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
        s' : Set (ULift.{w', w} M)
        h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
        h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
        hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
        h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
        ⊢ LE.le (Cardinal.lift.{max w w', max u v} L.card) (Cardinal.lift.{max (max u  …
      -/
      rw [← lift_lift.{w', w}]
      /-
        case intro.refine_2
        L : FirstOrder.Language
        M : Type w
        inst✝¹ : Nonempty M
        inst✝ : L.Structure M
        s : Set M
        κ : Cardinal.{w'}
        h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
        h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
        s' : Set (ULift.{w', w} M)
        h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
        h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
        hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
        h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
        ⊢ LE.le (Cardinal.lift.{w, max w' u v} (Cardinal.lift.{w', max u v} L.card)) ( …
      -/
      refine _root_.trans (lift_le.{w}.2 h3) ?_
      /-
        case intro.refine_2
        L : FirstOrder.Language
        M : Type w
        inst✝¹ : Nonempty M
        inst✝ : L.Structure M
        s : Set M
        κ : Cardinal.{w'}
        h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
        h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
        s' : Set (ULift.{w', w} M)
        h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
        h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
        hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
        h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
        ⊢ LE.le (Cardinal.lift.{w, max (max u v) w'} (Cardinal.lift.{max u v, w'} κ))  …
      -/
      rw [lift_lift, ← lift_lift.{w, max u v}, ← hs', ← h, lift_lift]
      /-
        🎉 no goals
      -/
    /-
      case intro
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : Nonempty M
      inst✝ : L.Structure M
      s : Set M
      κ : Cardinal.{w'}
      h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
      h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
      s' : Set (ULift.{w', w} M)
      h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
      h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
      hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
      h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
      ⊢ LE.le Cardinal.aleph0 (Cardinal.lift.{max u v, w} (Cardinal.mk ↑(Union.union …
    -/
  · refine _root_.trans ?_ (lift_le.2 (mk_le_mk_of_subset Set.subset_union_right))
    /-
      case intro
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : Nonempty M
      inst✝ : L.Structure M
      s : Set M
      κ : Cardinal.{w'}
      h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
      h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
      s' : Set (ULift.{w', w} M)
      h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
      h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
      hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
      h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
      ⊢ LE.le Cardinal.aleph0 (Cardinal.lift.{max u v, w} (Cardinal.mk ↑(Set.image ( …
    -/
    rw [aleph0_le_lift, ← aleph0_le_lift, h]
    /-
      case intro
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : Nonempty M
      inst✝ : L.Structure M
      s : Set M
      κ : Cardinal.{w'}
      h3 : LE.le (Cardinal.lift.{w', max u v} L.card) (Cardinal.lift.{max u v, w'} κ)
      h4 : LE.le (Cardinal.lift.{w, w'} κ) (Cardinal.lift.{w', w} (Cardinal.mk M))
      s' : Set (ULift.{w', w} M)
      h2 : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.mk ↑s')
      h1 : LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
      hs' : Eq (Cardinal.mk ↑s') (Cardinal.lift.{w, w'} κ)
      h : Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Set.image (⇑Equiv.ulift) s'))) (C …
      ⊢ LE.le Cardinal.aleph0 (Cardinal.mk ↑s')
    -/
    exact h1
    /-
      🎉 no goals
    -/


