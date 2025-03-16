/-- A term `t` with variables indexed by `α` can be evaluated by giving a value to each variable. -/
def realize (v : α → M) : ∀ _t : L.Term α, M
  | var k => v k
  | func f ts => funMap f fun i => (ts i).realize v

/- Porting note: The equation lemma of `realize` is too strong; it simplifies terms like the LHS of
`realize_functions_apply₁`. Even `eqns` can't fix this. We removed `simp` attr from `realize` and
prepare new simp lemmas for `realize`. -/


@[simp]
theorem realize_var (v : α → M) (k) : realize v (var k : L.Term α) = v k := rfl


@[simp]
theorem realize_func (v : α → M) {n} (f : L.Functions n) (ts) :
    realize v (func f ts : L.Term α) = funMap f fun i => (ts i).realize v := rfl


@[simp]
theorem realize_relabel {t : L.Term α} {g : α → β} {v : β → M} :
    (t.relabel g).realize v = t.realize (v ∘ g) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    t : L.Term α
    g : α → β
    v : β → M
    ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.relabel g t …
  -/
  induction' t with _ n f ts ih
    /-
      case var
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      β : Type v'
      g : α → β
      v : β → M
      a✝ : α
      ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.relabel g ( …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      β : Type v'
      g : α → β
      v : β → M
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term α
      ih : ∀ (a : Fin n), Eq (FirstOrder.Language.Term.realize v (FirstOrder.Languag …
      ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.relabel g ( …
    -/
  · simp [ih]
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_liftAt {n n' m : ℕ} {t : L.Term (α ⊕ (Fin n))} {v : α ⊕ (Fin (n + n')) → M} :
    (t.liftAt n' m).realize v =
      t.realize (v ∘ Sum.map id fun i : Fin _ =>
        if ↑i < m then Fin.castAdd n' i else Fin.addNat i n') :=
  realize_relabel


@[simp]
theorem realize_constants {c : L.Constants} {v : α → M} : c.term.realize v = c :=
  funMap_eq_coe_constants


@[simp]
theorem realize_functions_apply₁ {f : L.Functions 1} {t : L.Term α} {v : α → M} :
    (f.apply₁ t).realize v = funMap f ![t.realize v] := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    f : L.Functions 1
    t : L.Term α
    v : α → M
    ⊢ Eq (FirstOrder.Language.Term.realize v (f.apply₁ t)) (FirstOrder.Language.St …
  -/
  rw [Functions.apply₁, Term.realize]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    f : L.Functions 1
    t : L.Term α
    v : α → M
    ⊢ Eq (FirstOrder.Language.Structure.funMap f fun i => FirstOrder.Language.Term …
  -/
  refine congr rfl (funext fun i => ?_)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    f : L.Functions 1
    t : L.Term α
    v : α → M
    i : Fin 1
    ⊢ Eq (FirstOrder.Language.Term.realize v (Matrix.vecCons t Matrix.vecEmpty i)) …
  -/
  simp only [Matrix.cons_val_fin_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_functions_apply₂ {f : L.Functions 2} {t₁ t₂ : L.Term α} {v : α → M} :
    (f.apply₂ t₁ t₂).realize v = funMap f ![t₁.realize v, t₂.realize v] := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    f : L.Functions 2
    t₁ t₂ : L.Term α
    v : α → M
    ⊢ Eq (FirstOrder.Language.Term.realize v (f.apply₂ t₁ t₂)) (FirstOrder.Languag …
  -/
  rw [Functions.apply₂, Term.realize]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    f : L.Functions 2
    t₁ t₂ : L.Term α
    v : α → M
    ⊢ Eq (FirstOrder.Language.Structure.funMap f fun i => FirstOrder.Language.Term …
  -/
  refine congr rfl (funext (Fin.cases ?_ ?_))
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      f : L.Functions 2
      t₁ t₂ : L.Term α
      v : α → M
      ⊢ Eq (FirstOrder.Language.Term.realize v (Matrix.vecCons t₁ (Matrix.vecCons t₂ …
    -/
  · simp only [Matrix.cons_val_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      f : L.Functions 2
      t₁ t₂ : L.Term α
      v : α → M
      ⊢ ∀ (i : Fin 1), Eq (FirstOrder.Language.Term.realize v (Matrix.vecCons t₁ (Ma …
    -/
  · simp only [Matrix.cons_val_succ, Matrix.cons_val_fin_one, forall_const]
    /-
      🎉 no goals
    -/


theorem realize_con {A : Set M} {a : A} {v : α → M} : (L.con a).term.realize v = a :=
  rfl


@[simp]
theorem realize_subst {t : L.Term α} {tf : α → L.Term β} {v : β → M} :
    (t.subst tf).realize v = t.realize fun a => (tf a).realize v := by
  induction t with
  | var => rfl
  | func _ _ ih => simp [ih]


theorem realize_restrictVar [DecidableEq α] {t : L.Term α} {f : t.varFinset → β}
    {v : β → M} (v' : α → M) (hv' : ∀ a, v (f a) = v' a) :
     (t.restrictVar f).realize v = t.realize v' := by
  induction t with
  | var => simp [restrictVar, hv']
  | func _ _ ih =>
    exact congr rfl (funext fun i => ih i ((by simp [Function.comp_apply, hv'])))


/-- A special case of `realize_restrictVar`, included because we can add the `simp` attribute
to it -/
@[simp]
theorem realize_restrictVar' [DecidableEq α] {t : L.Term α} {s : Set α} (h : ↑t.varFinset ⊆ s)
    {v : α → M} : (t.restrictVar (Set.inclusion h)).realize (v ∘ (↑)) = t.realize v :=
                            /-
                              L : FirstOrder.Language
                              M : Type w
                              inst✝¹ : L.Structure M
                              α : Type u'
                              inst✝ : DecidableEq α
                              t : L.Term α
                              s : Set α
                              h : HasSubset.Subset (↑t.varFinset) s
                              v : α → M
                              ⊢ ∀ (a : Subtype fun x => Membership.mem t.varFinset x), Eq (Function.comp v S …
                            -/
  realize_restrictVar _ (by simp)
                            /-
                              🎉 no goals
                            -/


theorem realize_restrictVarLeft [DecidableEq α] {γ : Type*} {t : L.Term (α ⊕ γ)}
    {f : t.varFinsetLeft → β}
    {xs : β ⊕ γ → M} (xs' : α → M) (hxs' : ∀ a, xs (Sum.inl (f a)) = xs' a) :
    (t.restrictVarLeft f).realize xs = t.realize (Sum.elim xs' (xs ∘ Sum.inr)) := by
  induction t with
  | var a => cases a <;> simp [restrictVarLeft, hxs']
  | func _ _ ih =>
    exact congr rfl (funext fun i => ih i (by simp [hxs']))


/-- A special case of `realize_restrictVarLeft`, included because we can add the `simp` attribute
to it -/
@[simp]
theorem realize_restrictVarLeft' [DecidableEq α] {γ : Type*} {t : L.Term (α ⊕ γ)} {s : Set α}
    (h : ↑t.varFinsetLeft ⊆ s) {v : α → M} {xs : γ → M} :
    (t.restrictVarLeft (Set.inclusion h)).realize (Sum.elim (v ∘ (↑)) xs) =
      t.realize (Sum.elim v xs) :=
                                /-
                                  L : FirstOrder.Language
                                  M : Type w
                                  inst✝¹ : L.Structure M
                                  α : Type u'
                                  inst✝ : DecidableEq α
                                  γ : Type u_4
                                  t : L.Term (Sum α γ)
                                  s : Set α
                                  h : HasSubset.Subset (↑t.varFinsetLeft) s
                                  v : α → M
                                  xs : γ → M
                                  ⊢ ∀ (a : Subtype fun x => Membership.mem t.varFinsetLeft x), Eq (Sum.elim (Fun …
                                -/
  realize_restrictVarLeft _ (by simp)
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem realize_constantsToVars [L[[α]].Structure M] [(lhomWithConstants L α).IsExpansionOn M]
    {t : L[[α]].Term β} {v : β → M} :
    t.constantsToVars.realize (Sum.elim (fun a => ↑(L.con a)) v) = t.realize v := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    t : (L.withConstants α).Term β
    v : β → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) t.co …
  -/
  induction' t with _ n f ts ih
    /-
      case var
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      v : β → M
      a✝ : β
      ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      v : β → M
      n : Nat
      f : (L.withConstants α).Functions n
      ts : Fin n → (L.withConstants α).Term β
      ih : ∀ (a : Fin n), Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑ …
      ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
    -/
  · cases n
      /-
        case func.zero
        L : FirstOrder.Language
        M : Type w
        inst✝² : L.Structure M
        α : Type u'
        β : Type v'
        inst✝¹ : (L.withConstants α).Structure M
        inst✝ : (L.lhomWithConstants α).IsExpansionOn M
        v : β → M
        f : (L.withConstants α).Functions 0
        ts : Fin 0 → (L.withConstants α).Term β
        ih : ∀ (a : Fin 0), Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑ …
        ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
      -/
    · cases f
        /-
          case func.zero.inl
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          ts : Fin 0 → (L.withConstants α).Term β
          ih : ∀ (a : Fin 0), Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑ …
          val✝ : L.Functions 0
          ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
        -/
      · simp only [realize, ih, constantsOn, constantsOnFunc, constantsToVars]
        -- Porting note: below lemma does not work with simp for some reason
        /-
          case func.zero.inl
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          ts : Fin 0 → (L.withConstants α).Term β
          ih : ∀ (a : Fin 0), Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑ …
          val✝ : L.Functions 0
          ⊢ Eq (FirstOrder.Language.Structure.funMap val✝ fun i => FirstOrder.Language.T …
        -/
        rw [withConstants_funMap_sum_inl]
        /-
          🎉 no goals
        -/
        /-
          case func.zero.inr
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          ts : Fin 0 → (L.withConstants α).Term β
          ih : ∀ (a : Fin 0), Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑ …
          val✝ : (FirstOrder.Language.constantsOn α).Functions 0
          ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
        -/
      · simp only [realize, constantsToVars, Sum.elim_inl, funMap_eq_coe_constants]
        /-
          case func.zero.inr
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          ts : Fin 0 → (L.withConstants α).Term β
          ih : ∀ (a : Fin 0), Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑ …
          val✝ : (FirstOrder.Language.constantsOn α).Functions 0
          ⊢ Eq ↑(L.con val✝) ↑(Sum.inr val✝)
        -/
        rfl
        /-
          🎉 no goals
        -/
      /-
        case func.succ
        L : FirstOrder.Language
        M : Type w
        inst✝² : L.Structure M
        α : Type u'
        β : Type v'
        inst✝¹ : (L.withConstants α).Structure M
        inst✝ : (L.lhomWithConstants α).IsExpansionOn M
        v : β → M
        n✝ : Nat
        f : (L.withConstants α).Functions (HAdd.hAdd n✝ 1)
        ts : Fin (HAdd.hAdd n✝ 1) → (L.withConstants α).Term β
        ih : ∀ (a : Fin (HAdd.hAdd n✝ 1)), Eq (FirstOrder.Language.Term.realize (Sum.e …
        ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
      -/
    · cases' f with _ f
        /-
          case func.succ.inl
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          n✝ : Nat
          ts : Fin (HAdd.hAdd n✝ 1) → (L.withConstants α).Term β
          ih : ∀ (a : Fin (HAdd.hAdd n✝ 1)), Eq (FirstOrder.Language.Term.realize (Sum.e …
          val✝ : L.Functions (HAdd.hAdd n✝ 1)
          ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
        -/
      · simp only [realize, ih, constantsOn, constantsOnFunc, constantsToVars]
        -- Porting note: below lemma does not work with simp for some reason
        /-
          case func.succ.inl
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          n✝ : Nat
          ts : Fin (HAdd.hAdd n✝ 1) → (L.withConstants α).Term β
          ih : ∀ (a : Fin (HAdd.hAdd n✝ 1)), Eq (FirstOrder.Language.Term.realize (Sum.e …
          val✝ : L.Functions (HAdd.hAdd n✝ 1)
          ⊢ Eq (FirstOrder.Language.Structure.funMap val✝ fun i => FirstOrder.Language.T …
        -/
        rw [withConstants_funMap_sum_inl]
        /-
          🎉 no goals
        -/
        /-
          case func.succ.inr
          L : FirstOrder.Language
          M : Type w
          inst✝² : L.Structure M
          α : Type u'
          β : Type v'
          inst✝¹ : (L.withConstants α).Structure M
          inst✝ : (L.lhomWithConstants α).IsExpansionOn M
          v : β → M
          n✝ : Nat
          ts : Fin (HAdd.hAdd n✝ 1) → (L.withConstants α).Term β
          ih : ∀ (a : Fin (HAdd.hAdd n✝ 1)), Eq (FirstOrder.Language.Term.realize (Sum.e …
          f : (FirstOrder.Language.constantsOn α).Functions (HAdd.hAdd n✝ 1)
          ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim (fun a => ↑(L.con a)) v) (Fir …
        -/
      · exact isEmptyElim f
        /-
          🎉 no goals
        -/


@[simp]
theorem realize_varsToConstants [L[[α]].Structure M] [(lhomWithConstants L α).IsExpansionOn M]
    {t : L.Term (α ⊕ β)} {v : β → M} :
    t.varsToConstants.realize v = t.realize (Sum.elim (fun a => ↑(L.con a)) v) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    t : L.Term (Sum α β)
    v : β → M
    ⊢ Eq (FirstOrder.Language.Term.realize v t.varsToConstants) (FirstOrder.Langua …
  -/
  induction' t with ab n f ts ih
    /-
      case var
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      v : β → M
      ab : Sum α β
      ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.var ab).var …
    -/
  · cases' ab with a b
    -- Porting note: both cases were `simp [Language.con]`
      /-
        case var.inl
        L : FirstOrder.Language
        M : Type w
        inst✝² : L.Structure M
        α : Type u'
        β : Type v'
        inst✝¹ : (L.withConstants α).Structure M
        inst✝ : (L.lhomWithConstants α).IsExpansionOn M
        v : β → M
        a : α
        ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.var (Sum.in …
      -/
    · simp [Language.con, realize, funMap_eq_coe_constants]
      /-
        🎉 no goals
      -/
      /-
        case var.inr
        L : FirstOrder.Language
        M : Type w
        inst✝² : L.Structure M
        α : Type u'
        β : Type v'
        inst✝¹ : (L.withConstants α).Structure M
        inst✝ : (L.lhomWithConstants α).IsExpansionOn M
        v : β → M
        b : β
        ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.var (Sum.in …
      -/
    · simp [realize, constantMap]
      /-
        🎉 no goals
      -/
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      v : β → M
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term (Sum α β)
      ih : ∀ (a : Fin n), Eq (FirstOrder.Language.Term.realize v (ts a).varsToConsta …
      ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Language.Term.func f ts). …
    -/
  · simp only [realize, constantsOn, constantsOnFunc, ih, varsToConstants]
    -- Porting note: below lemma does not work with simp for some reason
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      v : β → M
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term (Sum α β)
      ih : ∀ (a : Fin n), Eq (FirstOrder.Language.Term.realize v (ts a).varsToConsta …
      ⊢ Eq (FirstOrder.Language.Structure.funMap (Sum.inl f) fun i => FirstOrder.Lan …
    -/
    rw [withConstants_funMap_sum_inl]
    /-
      🎉 no goals
    -/


theorem realize_constantsVarsEquivLeft [L[[α]].Structure M]
    [(lhomWithConstants L α).IsExpansionOn M] {n} {t : L[[α]].Term (β ⊕ (Fin n))} {v : β → M}
    {xs : Fin n → M} :
    (constantsVarsEquivLeft t).realize (Sum.elim (Sum.elim (fun a => ↑(L.con a)) v) xs) =
      t.realize (Sum.elim v xs) := by
  simp only [constantsVarsEquivLeft, realize_relabel, Equiv.coe_trans, Function.comp_apply,
    constantsVarsEquiv_apply, relabelEquiv_symm_apply]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    n : Nat
    t : (L.withConstants α).Term (Sum β (Fin n))
    v : β → M
    xs : Fin n → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (Sum.elim (Sum.elim (fun …
  -/
  refine _root_.trans ?_ realize_constantsToVars
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    n : Nat
    t : (L.withConstants α).Term (Sum β (Fin n))
    v : β → M
    xs : Fin n → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (Sum.elim (Sum.elim (fun …
  -/
  rcongr x
  /-
    case e_v.h
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    n : Nat
    t : (L.withConstants α).Term (Sum β (Fin n))
    v : β → M
    xs : Fin n → M
    x : Sum α (Sum β (Fin n))
    ⊢ Eq (Function.comp (Sum.elim (Sum.elim (fun a => ↑(L.con a)) v) xs) (⇑(Equiv. …
  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  rcases x with (a | (b | i)) <;> simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem realize_onTerm [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M] (t : L.Term α)
    (v : α → M) : (φ.onTerm t).realize v = t.realize v := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : L'.Structure M
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    t : L.Term α
    v : α → M
    ⊢ Eq (FirstOrder.Language.Term.realize v (φ.onTerm t)) (FirstOrder.Language.Te …
  -/
  induction' t with _ n f ts ih
    /-
      case var
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      inst✝¹ : L'.Structure M
      φ : L.LHom L'
      inst✝ : φ.IsExpansionOn M
      v : α → M
      a✝ : α
      ⊢ Eq (FirstOrder.Language.Term.realize v (φ.onTerm (FirstOrder.Language.Term.v …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case func
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      inst✝¹ : L'.Structure M
      φ : L.LHom L'
      inst✝ : φ.IsExpansionOn M
      v : α → M
      n : Nat
      f : L.Functions n
      ts : Fin n → L.Term α
      ih : ∀ (a : Fin n), Eq (FirstOrder.Language.Term.realize v (φ.onTerm (ts a)))  …
      ⊢ Eq (FirstOrder.Language.Term.realize v (φ.onTerm (FirstOrder.Language.Term.f …
    -/
  · simp only [Term.realize, LHom.onTerm, LHom.map_onFunction, ih]
    /-
      🎉 no goals
    -/


@[simp]
theorem HomClass.realize_term {F : Type*} [FunLike F M N] [HomClass L F M N]
    (g : F) {t : L.Term α} {v : α → M} :
    t.realize (g ∘ v) = g (t.realize v) := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝³ : L.Structure M
    inst✝² : L.Structure N
    α : Type u'
    F : Type u_4
    inst✝¹ : FunLike F M N
    inst✝ : L.HomClass F M N
    g : F
    t : L.Term α
    v : α → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (⇑g) v) t) (g (FirstOrde …
  -/
  induction t
    /-
      case var
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      α : Type u'
      F : Type u_4
      inst✝¹ : FunLike F M N
      inst✝ : L.HomClass F M N
      g : F
      v : α → M
      a✝ : α
      ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (⇑g) v) (FirstOrder.Lang …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      α : Type u'
      F : Type u_4
      inst✝¹ : FunLike F M N
      inst✝ : L.HomClass F M N
      g : F
      v : α → M
      l✝ : Nat
      _f✝ : L.Functions l✝
      _ts✝ : Fin l✝ → L.Term α
      _ts_ih✝ : ∀ (a : Fin l✝), Eq (FirstOrder.Language.Term.realize (Function.comp  …
      ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (⇑g) v) (FirstOrder.Lang …
    -/
  · rw [Term.realize, Term.realize, HomClass.map_fun]
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      α : Type u'
      F : Type u_4
      inst✝¹ : FunLike F M N
      inst✝ : L.HomClass F M N
      g : F
      v : α → M
      l✝ : Nat
      _f✝ : L.Functions l✝
      _ts✝ : Fin l✝ → L.Term α
      _ts_ih✝ : ∀ (a : Fin l✝), Eq (FirstOrder.Language.Term.realize (Function.comp  …
      ⊢ Eq (FirstOrder.Language.Structure.funMap _f✝ fun i => FirstOrder.Language.Te …
    -/
    refine congr rfl ?_
    /-
      case func
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      α : Type u'
      F : Type u_4
      inst✝¹ : FunLike F M N
      inst✝ : L.HomClass F M N
      g : F
      v : α → M
      l✝ : Nat
      _f✝ : L.Functions l✝
      _ts✝ : Fin l✝ → L.Term α
      _ts_ih✝ : ∀ (a : Fin l✝), Eq (FirstOrder.Language.Term.realize (Function.comp  …
      ⊢ Eq (fun i => FirstOrder.Language.Term.realize (Function.comp (⇑g) v) (_ts✝ i …
    -/
    ext x
    /-
      case func.h
      L : FirstOrder.Language
      M : Type w
      N : Type u_1
      inst✝³ : L.Structure M
      inst✝² : L.Structure N
      α : Type u'
      F : Type u_4
      inst✝¹ : FunLike F M N
      inst✝ : L.HomClass F M N
      g : F
      v : α → M
      l✝ : Nat
      _f✝ : L.Functions l✝
      _ts✝ : Fin l✝ → L.Term α
      _ts_ih✝ : ∀ (a : Fin l✝), Eq (FirstOrder.Language.Term.realize (Function.comp  …
      x : Fin l✝
      ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (⇑g) v) (_ts✝ x)) (Funct …
    -/
    simp [*]
    /-
      🎉 no goals
    -/


/-- A bounded formula can be evaluated as true or false by giving values to each free variable. -/
def Realize : ∀ {l} (_f : L.BoundedFormula α l) (_v : α → M) (_xs : Fin l → M), Prop
  | _, falsum, _v, _xs => False
  | _, equal t₁ t₂, v, xs => t₁.realize (Sum.elim v xs) = t₂.realize (Sum.elim v xs)
  | _, rel R ts, v, xs => RelMap R fun i => (ts i).realize (Sum.elim v xs)
  | _, imp f₁ f₂, v, xs => Realize f₁ v xs → Realize f₂ v xs
  | _, all f, v, xs => ∀ x : M, Realize f v (snoc xs x)


@[simp]
theorem realize_bot : (⊥ : L.BoundedFormula α l).Realize v xs ↔ False :=
  Iff.rfl


@[simp]
theorem realize_not : φ.not.Realize v xs ↔ ¬φ.Realize v xs :=
  Iff.rfl


@[simp]
theorem realize_bdEqual (t₁ t₂ : L.Term (α ⊕ (Fin l))) :
    (t₁.bdEqual t₂).Realize v xs ↔ t₁.realize (Sum.elim v xs) = t₂.realize (Sum.elim v xs) :=
  Iff.rfl


@[simp]
                                                                           /-
                                                                             L : FirstOrder.Language
                                                                             M : Type w
                                                                             inst✝ : L.Structure M
                                                                             α : Type u'
                                                                             l : Nat
                                                                             v : α → M
                                                                             xs : Fin l → M
                                                                             ⊢ Iff (Top.top.Realize v xs) True
                                                                           -/
theorem realize_top : (⊤ : L.BoundedFormula α l).Realize v xs ↔ True := by simp [Top.top]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem realize_inf : (φ ⊓ ψ).Realize v xs ↔ φ.Realize v xs ∧ ψ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    φ ψ : L.BoundedFormula α l
    v : α → M
    xs : Fin l → M
    ⊢ Iff ((Min.min φ ψ).Realize v xs) (And (φ.Realize v xs) (ψ.Realize v xs))
  -/
  simp [Inf.inf, Realize]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_foldr_inf (l : List (L.BoundedFormula α n)) (v : α → M) (xs : Fin n → M) :
    (l.foldr (· ⊓ ·) ⊤).Realize v xs ↔ ∀ φ ∈ l, BoundedFormula.Realize φ v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    l : List (L.BoundedFormula α n)
    v : α → M
    xs : Fin n → M
    ⊢ Iff ((List.foldr (fun x1 x2 => Min.min x1 x2) Top.top l).Realize v xs) (∀ (φ …
  -/
  induction' l with φ l ih
    /-
      case nil
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n : Nat
      v : α → M
      xs : Fin n → M
      ⊢ Iff ((List.foldr (fun x1 x2 => Min.min x1 x2) Top.top List.nil).Realize v xs …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n : Nat
      v : α → M
      xs : Fin n → M
      φ : L.BoundedFormula α n
      l : List (L.BoundedFormula α n)
      ih : Iff ((List.foldr (fun x1 x2 => Min.min x1 x2) Top.top l).Realize v xs) (∀ …
      ⊢ Iff ((List.foldr (fun x1 x2 => Min.min x1 x2) Top.top (List.cons φ l)).Reali …
    -/
  · simp [ih]
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_imp : (φ.imp ψ).Realize v xs ↔ φ.Realize v xs → ψ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    φ ψ : L.BoundedFormula α l
    v : α → M
    xs : Fin l → M
    ⊢ Iff ((φ.imp ψ).Realize v xs) (φ.Realize v xs → ψ.Realize v xs)
  -/
  simp only [Realize]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_rel {k : ℕ} {R : L.Relations k} {ts : Fin k → L.Term _} :
    (R.boundedFormula ts).Realize v xs ↔ RelMap R fun i => (ts i).realize (Sum.elim v xs) :=
  Iff.rfl


@[simp]
theorem realize_rel₁ {R : L.Relations 1} {t : L.Term _} :
    (R.boundedFormula₁ t).Realize v xs ↔ RelMap R ![t.realize (Sum.elim v xs)] := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    v : α → M
    xs : Fin l → M
    R : L.Relations 1
    t : L.Term (Sum α (Fin l))
    ⊢ Iff ((R.boundedFormula₁ t).Realize v xs) (FirstOrder.Language.Structure.RelM …
  -/
  rw [Relations.boundedFormula₁, realize_rel, iff_eq_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    v : α → M
    xs : Fin l → M
    R : L.Relations 1
    t : L.Term (Sum α (Fin l))
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R fun i => FirstOrder.Language.Term …
  -/
  refine congr rfl (funext fun _ => ?_)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    v : α → M
    xs : Fin l → M
    R : L.Relations 1
    t : L.Term (Sum α (Fin l))
    x✝ : Fin 1
    ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v xs) (Matrix.vecCons t Matri …
  -/
  simp only [Matrix.cons_val_fin_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_rel₂ {R : L.Relations 2} {t₁ t₂ : L.Term _} :
    (R.boundedFormula₂ t₁ t₂).Realize v xs ↔
      RelMap R ![t₁.realize (Sum.elim v xs), t₂.realize (Sum.elim v xs)] := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    v : α → M
    xs : Fin l → M
    R : L.Relations 2
    t₁ t₂ : L.Term (Sum α (Fin l))
    ⊢ Iff ((R.boundedFormula₂ t₁ t₂).Realize v xs) (FirstOrder.Language.Structure. …
  -/
  rw [Relations.boundedFormula₂, realize_rel, iff_eq_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    v : α → M
    xs : Fin l → M
    R : L.Relations 2
    t₁ t₂ : L.Term (Sum α (Fin l))
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R fun i => FirstOrder.Language.Term …
  -/
  refine congr rfl (funext (Fin.cases ?_ ?_))
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      l : Nat
      v : α → M
      xs : Fin l → M
      R : L.Relations 2
      t₁ t₂ : L.Term (Sum α (Fin l))
      ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v xs) (Matrix.vecCons t₁ (Mat …
    -/
  · simp only [Matrix.cons_val_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      l : Nat
      v : α → M
      xs : Fin l → M
      R : L.Relations 2
      t₁ t₂ : L.Term (Sum α (Fin l))
      ⊢ ∀ (i : Fin 1), Eq (FirstOrder.Language.Term.realize (Sum.elim v xs) (Matrix. …
    -/
  · simp only [Matrix.cons_val_succ, Matrix.cons_val_fin_one, forall_const]
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_sup : (φ ⊔ ψ).Realize v xs ↔ φ.Realize v xs ∨ ψ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    φ ψ : L.BoundedFormula α l
    v : α → M
    xs : Fin l → M
    ⊢ Iff ((Max.max φ ψ).Realize v xs) (Or (φ.Realize v xs) (ψ.Realize v xs))
  -/
  simp only [realize, max, realize_not, eq_iff_iff]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    φ ψ : L.BoundedFormula α l
    v : α → M
    xs : Fin l → M
    ⊢ Iff ((φ.not.imp ψ).Realize v xs) (Or (φ.Realize v xs) (ψ.Realize v xs))
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_foldr_sup (l : List (L.BoundedFormula α n)) (v : α → M) (xs : Fin n → M) :
    (l.foldr (· ⊔ ·) ⊥).Realize v xs ↔ ∃ φ ∈ l, BoundedFormula.Realize φ v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    l : List (L.BoundedFormula α n)
    v : α → M
    xs : Fin n → M
    ⊢ Iff ((List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot l).Realize v xs) (Exis …
  -/
  induction' l with φ l ih
    /-
      case nil
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n : Nat
      v : α → M
      xs : Fin n → M
      ⊢ Iff ((List.foldr (fun x1 x2 => Max.max x1 x2) Bot.bot List.nil).Realize v xs …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp_rw [List.foldr_cons, realize_sup, ih, List.mem_cons, or_and_right, exists_or,
      exists_eq_left]


@[simp]
theorem realize_all : (all θ).Realize v xs ↔ ∀ a : M, θ.Realize v (Fin.snoc xs a) :=
  Iff.rfl


@[simp]
theorem realize_ex : θ.ex.Realize v xs ↔ ∃ a : M, θ.Realize v (Fin.snoc xs a) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    θ : L.BoundedFormula α l.succ
    v : α → M
    xs : Fin l → M
    ⊢ Iff (θ.ex.Realize v xs) (Exists fun a => θ.Realize v (Fin.snoc xs a))
  -/
  rw [BoundedFormula.ex, realize_not, realize_all, not_forall]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    θ : L.BoundedFormula α l.succ
    v : α → M
    xs : Fin l → M
    ⊢ Iff (Exists fun x => Not (θ.not.Realize v (Fin.snoc xs x))) (Exists fun a => …
  -/
  simp_rw [realize_not, Classical.not_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_iff : (φ.iff ψ).Realize v xs ↔ (φ.Realize v xs ↔ ψ.Realize v xs) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    l : Nat
    φ ψ : L.BoundedFormula α l
    v : α → M
    xs : Fin l → M
    ⊢ Iff ((φ.iff ψ).Realize v xs) (Iff (φ.Realize v xs) (ψ.Realize v xs))
  -/
  simp only [BoundedFormula.iff, realize_inf, realize_imp, and_imp, ← iff_def]
  /-
    🎉 no goals
  -/


theorem realize_castLE_of_eq {m n : ℕ} (h : m = n) {h' : m ≤ n} {φ : L.BoundedFormula α m}
    {v : α → M} {xs : Fin n → M} : (φ.castLE h').Realize v xs ↔ φ.Realize v (xs ∘ cast h) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    m n : Nat
    h : Eq m n
    h' : LE.le m n
    φ : L.BoundedFormula α m
    v : α → M
    xs : Fin n → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.castLE h' φ).Realize v xs) (φ.Reali …
  -/
  subst h
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    m : Nat
    φ : L.BoundedFormula α m
    v : α → M
    h' : LE.le m m
    xs : Fin m → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.castLE h' φ).Realize v xs) (φ.Reali …
  -/
  simp only [castLE_rfl, cast_refl, OrderIso.coe_refl, Function.comp_id]
  /-
    🎉 no goals
  -/


theorem realize_mapTermRel_id [L'.Structure M]
    {ft : ∀ n, L.Term (α ⊕ (Fin n)) → L'.Term (β ⊕ (Fin n))}
    {fr : ∀ n, L.Relations n → L'.Relations n} {n} {φ : L.BoundedFormula α n} {v : α → M}
    {v' : β → M} {xs : Fin n → M}
    (h1 :
      ∀ (n) (t : L.Term (α ⊕ (Fin n))) (xs : Fin n → M),
        (ft n t).realize (Sum.elim v' xs) = t.realize (Sum.elim v xs))
    (h2 : ∀ (n) (R : L.Relations n) (x : Fin n → M), RelMap (fr n R) x = RelMap R x) :
    (φ.mapTermRel ft fr fun _ => id).Realize v' xs ↔ φ.Realize v xs := by
  induction φ with
  | falsum => rfl
  | equal => simp [mapTermRel, Realize, h1]
  | rel => simp [mapTermRel, Realize, h1, h2]
  | imp _ _ ih1 ih2 => simp [mapTermRel, Realize, ih1, ih2]
  | all _ ih => simp only [mapTermRel, Realize, ih, id]


theorem realize_mapTermRel_add_castLe [L'.Structure M] {k : ℕ}
    {ft : ∀ n, L.Term (α ⊕ (Fin n)) → L'.Term (β ⊕ (Fin (k + n)))}
    {fr : ∀ n, L.Relations n → L'.Relations n} {n} {φ : L.BoundedFormula α n}
    (v : ∀ {n}, (Fin (k + n) → M) → α → M) {v' : β → M} (xs : Fin (k + n) → M)
    (h1 :
      ∀ (n) (t : L.Term (α ⊕ (Fin n))) (xs' : Fin (k + n) → M),
        (ft n t).realize (Sum.elim v' xs') = t.realize (Sum.elim (v xs') (xs' ∘ Fin.natAdd _)))
    (h2 : ∀ (n) (R : L.Relations n) (x : Fin n → M), RelMap (fr n R) x = RelMap R x)
    (hv : ∀ (n) (xs : Fin (k + n) → M) (x : M), @v (n + 1) (snoc xs x : Fin _ → M) = v xs) :
    (φ.mapTermRel ft fr fun _ => castLE (add_assoc _ _ _).symm.le).Realize v' xs ↔
      φ.Realize (v xs) (xs ∘ Fin.natAdd _) := by
  induction φ with
  | falsum => rfl
  | equal => simp [mapTermRel, Realize, h1]
  | rel => simp [mapTermRel, Realize, h1, h2]
  | imp _ _ ih1 ih2 => simp [mapTermRel, Realize, ih1, ih2]
  | all _ ih => simp [mapTermRel, Realize, ih, hv]


@[simp]
theorem realize_relabel {m n : ℕ} {φ : L.BoundedFormula α n} {g : α → β ⊕ (Fin m)} {v : β → M}
    {xs : Fin (m + n) → M} :
    (φ.relabel g).Realize v xs ↔
      φ.Realize (Sum.elim v (xs ∘ Fin.castAdd n) ∘ g) (xs ∘ Fin.natAdd m) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    m n : Nat
    φ : L.BoundedFormula α n
    g : α → Sum β (Fin m)
    v : β → M
    xs : Fin (HAdd.hAdd m n) → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.relabel g φ).Realize v xs) (φ.Reali …
  -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  apply realize_mapTermRel_add_castLe <;> simp
                                          /-
                                            🎉 no goals
                                          -/


theorem realize_liftAt {n n' m : ℕ} {φ : L.BoundedFormula α n} {v : α → M} {xs : Fin (n + n') → M}
    (hmn : m + n' ≤ n + 1) :
    (φ.liftAt n' m).Realize v xs ↔
      φ.Realize v (xs ∘ fun i => if ↑i < m then Fin.castAdd n' i else Fin.addNat i n') := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n n' m : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin (HAdd.hAdd n n') → M
    hmn : LE.le (HAdd.hAdd m n') (HAdd.hAdd n 1)
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.liftAt n' m φ).Realize v xs) (φ.Rea …
  -/
  rw [liftAt]
  induction φ with
  | falsum => simp [mapTermRel, Realize]
  | equal => simp [mapTermRel, Realize, realize_rel, realize_liftAt, Sum.elim_comp_map]
  | rel => simp [mapTermRel, Realize, realize_rel, realize_liftAt, Sum.elim_comp_map]
  | imp _ _ ih1 ih2 => simp only [mapTermRel, Realize, ih1 hmn, ih2 hmn]
  | @all k _ ih3 =>
    have h : k + 1 + n' = k + n' + 1 := by rw [add_assoc, add_comm 1 n', ← add_assoc]
    simp only [mapTermRel, Realize, realize_castLE_of_eq h, ih3 (hmn.trans k.succ.le_succ)]
    refine forall_congr' fun x => iff_eq_eq.mpr (congr rfl (funext (Fin.lastCases ?_ fun i => ?_)))
    · simp only [Function.comp_apply, val_last, snoc_last]
      refine (congr rfl (Fin.ext ?_)).trans (snoc_last _ _)
      split_ifs <;> dsimp; omega
    · simp only [Function.comp_apply, Fin.snoc_castSucc]
      refine (congr rfl (Fin.ext ?_)).trans (snoc_castSucc _ _ _)
      simp only [coe_castSucc, coe_cast]
      split_ifs <;> simp


theorem realize_liftAt_one {n m : ℕ} {φ : L.BoundedFormula α n} {v : α → M} {xs : Fin (n + 1) → M}
    (hmn : m ≤ n) :
    (φ.liftAt 1 m).Realize v xs ↔
      φ.Realize v (xs ∘ fun i => if ↑i < m then castSucc i else i.succ) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n m : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin (HAdd.hAdd n 1) → M
    hmn : LE.le m n
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.liftAt 1 m φ).Realize v xs) (φ.Real …
  -/
  simp [realize_liftAt (add_le_add_right hmn 1), castSucc]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_liftAt_one_self {n : ℕ} {φ : L.BoundedFormula α n} {v : α → M}
    {xs : Fin (n + 1) → M} : (φ.liftAt 1 n).Realize v xs ↔ φ.Realize v (xs ∘ castSucc) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin (HAdd.hAdd n 1) → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.liftAt 1 n φ).Realize v xs) (φ.Real …
  -/
  rw [realize_liftAt_one (refl n), iff_eq_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin (HAdd.hAdd n 1) → M
    ⊢ Eq (φ.Realize v (Function.comp xs fun i => ite (LT.lt (↑i) n) i.castSucc i.s …
  -/
  refine congr rfl (congr rfl (funext fun i => ?_))
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin (HAdd.hAdd n 1) → M
    i : Fin n
    ⊢ Eq (ite (LT.lt (↑i) n) i.castSucc i.succ) i.castSucc
  -/
  rw [if_pos i.is_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_subst {φ : L.BoundedFormula α n} {tf : α → L.Term β} {v : β → M} {xs : Fin n → M} :
    (φ.subst tf).Realize v xs ↔ φ.Realize (fun a => (tf a).realize v) xs :=
  realize_mapTermRel_id
    (fun n t x => by
      /-
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        β : Type v'
        n✝ : Nat
        φ : L.BoundedFormula α n✝
        tf : α → L.Term β
        v : β → M
        xs : Fin n✝ → M
        n : Nat
        t : L.Term (Sum α (Fin n))
        x : Fin n → M
        ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v x) (t.subst (Sum.elim (Func …
      -/
      rw [Term.realize_subst]
      /-
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        β : Type v'
        n✝ : Nat
        φ : L.BoundedFormula α n✝
        tf : α → L.Term β
        v : β → M
        xs : Fin n✝ → M
        n : Nat
        t : L.Term (Sum α (Fin n))
        x : Fin n → M
        ⊢ Eq (FirstOrder.Language.Term.realize (fun a => FirstOrder.Language.Term.real …
      -/
      rcongr a
      /-
        case e_v.h
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        β : Type v'
        n✝ : Nat
        φ : L.BoundedFormula α n✝
        tf : α → L.Term β
        v : β → M
        xs : Fin n✝ → M
        n : Nat
        t : L.Term (Sum α (Fin n))
        x : Fin n → M
        a : Sum α (Fin n)
        ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v x) (Sum.elim (Function.comp …
      -/
      cases a
        /-
          case e_v.h.inl
          L : FirstOrder.Language
          M : Type w
          inst✝ : L.Structure M
          α : Type u'
          β : Type v'
          n✝ : Nat
          φ : L.BoundedFormula α n✝
          tf : α → L.Term β
          v : β → M
          xs : Fin n✝ → M
          n : Nat
          t : L.Term (Sum α (Fin n))
          x : Fin n → M
          val✝ : α
          ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v x) (Sum.elim (Function.comp …
        -/
      · simp only [Sum.elim_inl, Function.comp_apply, Term.realize_relabel, Sum.elim_comp_inl]
        /-
          🎉 no goals
        -/
        /-
          case e_v.h.inr
          L : FirstOrder.Language
          M : Type w
          inst✝ : L.Structure M
          α : Type u'
          β : Type v'
          n✝ : Nat
          φ : L.BoundedFormula α n✝
          tf : α → L.Term β
          v : β → M
          xs : Fin n✝ → M
          n : Nat
          t : L.Term (Sum α (Fin n))
          x : Fin n → M
          val✝ : Fin n
          ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v x) (Sum.elim (Function.comp …
        -/
      · rfl)
        /-
          🎉 no goals
        -/
        /-
          L : FirstOrder.Language
          M : Type w
          inst✝ : L.Structure M
          α : Type u'
          β : Type v'
          n : Nat
          φ : L.BoundedFormula α n
          tf : α → L.Term β
          v : β → M
          xs : Fin n → M
          ⊢ ∀ (n : Nat) (R : L.Relations n) (x : Fin n → M), Eq (FirstOrder.Language.Str …
        -/
    (by simp)
        /-
          🎉 no goals
        -/


theorem realize_restrictFreeVar [DecidableEq α] {n : ℕ} {φ : L.BoundedFormula α n}
    {f : φ.freeVarFinset → β} {v : β → M} {xs : Fin n → M}
    (v' : α → M) (hv' : ∀ a, v (f a) = v' a) :
    (φ.restrictFreeVar f).Realize v xs ↔ φ.Realize v' xs := by
  induction φ with
  | falsum => rfl
  | equal =>
    simp only [Realize, freeVarFinset.eq_2]
    rw [realize_restrictVarLeft v' (by simp [hv']), realize_restrictVarLeft v' (by simp [hv'])]
    simp [Function.comp_apply]
  | rel =>
    simp only [Realize, freeVarFinset.eq_3, Finset.biUnion_val, restrictFreeVar]
    congr!
    rw [realize_restrictVarLeft v' (by simp [hv'])]
    simp [Function.comp_apply]
  | imp _ _ ih1 ih2 =>
    simp only [Realize, freeVarFinset.eq_4]
    rw [ih1, ih2] <;> simp [hv']
  | all _ ih3 =>
    simp only [restrictFreeVar, Realize]
    refine forall_congr' (fun _ => ?_)
    rw [ih3]; simp [hv']


/-- A special case of `realize_restrictFreeVar`, included because we can add the `simp` attribute
to it -/
@[simp]
theorem realize_restrictFreeVar' [DecidableEq α] {n : ℕ} {φ : L.BoundedFormula α n} {s : Set α}
    (h : ↑φ.freeVarFinset ⊆ s) {v : α → M} {xs : Fin n → M} :
    (φ.restrictFreeVar (Set.inclusion h)).Realize (v ∘ (↑)) xs ↔ φ.Realize v xs :=
                                /-
                                  L : FirstOrder.Language
                                  M : Type w
                                  inst✝¹ : L.Structure M
                                  α : Type u'
                                  inst✝ : DecidableEq α
                                  n : Nat
                                  φ : L.BoundedFormula α n
                                  s : Set α
                                  h : HasSubset.Subset (↑φ.freeVarFinset) s
                                  v : α → M
                                  xs : Fin n → M
                                  ⊢ ∀ (a : Subtype fun x => Membership.mem φ.freeVarFinset x), Eq (Function.comp …
                                -/
  realize_restrictFreeVar _ (by simp)
                                /-
                                  🎉 no goals
                                -/


theorem realize_constantsVarsEquiv [L[[α]].Structure M] [(lhomWithConstants L α).IsExpansionOn M]
    {n} {φ : L[[α]].BoundedFormula β n} {v : β → M} {xs : Fin n → M} :
    (constantsVarsEquiv φ).Realize (Sum.elim (fun a => ↑(L.con a)) v) xs ↔ φ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    n : Nat
    φ : (L.withConstants α).BoundedFormula β n
    v : β → M
    xs : Fin n → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.constantsVarsEquiv φ).Realize (Sum. …
  -/
  refine realize_mapTermRel_id (fun n t xs => realize_constantsVarsEquivLeft) fun n R xs => ?_
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [← (lhomWithConstants L α).map_onRelation
      (Equiv.sumEmpty (L.Relations n) ((constantsOn α).Relations n) R) xs]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    n✝ : Nat
    φ : (L.withConstants α).BoundedFormula β n✝
    v : β → M
    xs✝ : Fin n✝ → M
    n : Nat
    R : (L.withConstants α).Relations n
    xs : Fin n → M
    ⊢ Eq (FirstOrder.Language.Structure.RelMap ((L.lhomWithConstants α).onRelation …
  -/
  rcongr
  /-
    case a
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    β : Type v'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    n✝ : Nat
    φ : (L.withConstants α).BoundedFormula β n✝
    v : β → M
    xs✝ : Fin n✝ → M
    n : Nat
    R : (L.withConstants α).Relations n
    xs : Fin n → M
    ⊢ Iff (FirstOrder.Language.Structure.RelMap ((L.lhomWithConstants α).onRelatio …
  -/
  cases' R with R R
    /-
      case a.inl
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      n✝ : Nat
      φ : (L.withConstants α).BoundedFormula β n✝
      v : β → M
      xs✝ : Fin n✝ → M
      n : Nat
      xs : Fin n → M
      R : L.Relations n
      ⊢ Iff (FirstOrder.Language.Structure.RelMap ((L.lhomWithConstants α).onRelatio …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a.inr
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      β : Type v'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      n✝ : Nat
      φ : (L.withConstants α).BoundedFormula β n✝
      v : β → M
      xs✝ : Fin n✝ → M
      n : Nat
      xs : Fin n → M
      R : (FirstOrder.Language.constantsOn α).Relations n
      ⊢ Iff (FirstOrder.Language.Structure.RelMap ((L.lhomWithConstants α).onRelatio …
    -/
  · exact isEmptyElim R
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_relabelEquiv {g : α ≃ β} {k} {φ : L.BoundedFormula α k} {v : β → M}
    {xs : Fin k → M} : (relabelEquiv g φ).Realize v xs ↔ φ.Realize (v ∘ g) xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    g : _root_.Equiv α β
    k : Nat
    φ : L.BoundedFormula α k
    v : β → M
    xs : Fin k → M
    ⊢ Iff (((FirstOrder.Language.BoundedFormula.relabelEquiv g) φ).Realize v xs) ( …
  -/
  simp only [relabelEquiv, mapTermRelEquiv_apply, Equiv.coe_refl]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    g : _root_.Equiv α β
    k : Nat
    φ : L.BoundedFormula α k
    v : β → M
    xs : Fin k → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.mapTermRel (fun n => ⇑(FirstOrder.L …
  -/
  refine realize_mapTermRel_id (fun n t xs => ?_) fun _ _ _ => rfl
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    g : _root_.Equiv α β
    k : Nat
    φ : L.BoundedFormula α k
    v : β → M
    xs✝ : Fin k → M
    n : Nat
    t : L.Term (Sum α (Fin n))
    xs : Fin n → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Sum.elim v xs) ((FirstOrder.Language.T …
  -/
  simp only [relabelEquiv_apply, Term.realize_relabel]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    g : _root_.Equiv α β
    k : Nat
    φ : L.BoundedFormula α k
    v : β → M
    xs✝ : Fin k → M
    n : Nat
    t : L.Term (Sum α (Fin n))
    xs : Fin n → M
    ⊢ Eq (FirstOrder.Language.Term.realize (Function.comp (Sum.elim v xs) ⇑(g.sumC …
  -/
  refine congr (congr rfl ?_) rfl
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    g : _root_.Equiv α β
    k : Nat
    φ : L.BoundedFormula α k
    v : β → M
    xs✝ : Fin k → M
    n : Nat
    t : L.Term (Sum α (Fin n))
    xs : Fin n → M
    ⊢ Eq (Function.comp (Sum.elim v xs) ⇑(g.sumCongr (_root_.Equiv.refl (Fin n)))) …
  -/
                  /-
                    🎉 no goals
                  -/
  ext (i | i) <;> rfl
                  /-
                    🎉 no goals
                  -/


theorem realize_all_liftAt_one_self {n : ℕ} {φ : L.BoundedFormula α n} {v : α → M}
    {xs : Fin n → M} : (φ.liftAt 1 n).all.Realize v xs ↔ φ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    inst✝ : Nonempty M
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin n → M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.liftAt 1 n φ).all.Realize v xs) (φ. …
  -/
  inhabit M
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    inst✝ : Nonempty M
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin n → M
    inhabited_h : Inhabited M
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.liftAt 1 n φ).all.Realize v xs) (φ. …
  -/
  simp only [realize_all, realize_liftAt_one_self]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    inst✝ : Nonempty M
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    xs : Fin n → M
    inhabited_h : Inhabited M
    ⊢ Iff (∀ (a : M), φ.Realize v (Function.comp (Fin.snoc xs a) Fin.castSucc)) (φ …
  -/
  refine ⟨fun h => ?_, fun h a => ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      inst✝ : Nonempty M
      n : Nat
      φ : L.BoundedFormula α n
      v : α → M
      xs : Fin n → M
      inhabited_h : Inhabited M
      h : ∀ (a : M), φ.Realize v (Function.comp (Fin.snoc xs a) Fin.castSucc)
      ⊢ φ.Realize v xs
    -/
  · refine (congr rfl (funext fun i => ?_)).mp (h default)
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      inst✝ : Nonempty M
      n : Nat
      φ : L.BoundedFormula α n
      v : α → M
      xs : Fin n → M
      inhabited_h : Inhabited M
      h : ∀ (a : M), φ.Realize v (Function.comp (Fin.snoc xs a) Fin.castSucc)
      i : Fin n
      ⊢ Eq (Function.comp (Fin.snoc xs Inhabited.default) Fin.castSucc i) (xs i)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      inst✝ : Nonempty M
      n : Nat
      φ : L.BoundedFormula α n
      v : α → M
      xs : Fin n → M
      inhabited_h : Inhabited M
      h : φ.Realize v xs
      a : M
      ⊢ φ.Realize v (Function.comp (Fin.snoc xs a) Fin.castSucc)
    -/
  · refine (congr rfl (funext fun i => ?_)).mp h
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      inst✝ : Nonempty M
      n : Nat
      φ : L.BoundedFormula α n
      v : α → M
      xs : Fin n → M
      inhabited_h : Inhabited M
      h : φ.Realize v xs
      a : M
      i : Fin n
      ⊢ Eq (xs i) (Function.comp (Fin.snoc xs a) Fin.castSucc i)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_onBoundedFormula [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M] {n : ℕ}
    (ψ : L.BoundedFormula α n) {v : α → M} {xs : Fin n → M} :
    (φ.onBoundedFormula ψ).Realize v xs ↔ ψ.Realize v xs := by
  induction ψ with
  | falsum => rfl
  | equal => simp only [onBoundedFormula, realize_bdEqual, realize_onTerm]; rfl
  | rel =>
    simp only [onBoundedFormula, realize_rel, LHom.map_onRelation,
      Function.comp_apply, realize_onTerm]
    rfl
  | imp _ _ ih1 ih2 => simp only [onBoundedFormula, ih1, ih2, realize_imp]
  | all _ ih3 => simp only [onBoundedFormula, ih3, realize_all]


/-- A formula can be evaluated as true or false by giving values to each free variable. -/
nonrec def Realize (φ : L.Formula α) (v : α → M) : Prop :=
  φ.Realize v default


@[simp]
theorem realize_not : φ.not.Realize v ↔ ¬φ.Realize v :=
  Iff.rfl


@[simp]
theorem realize_bot : (⊥ : L.Formula α).Realize v ↔ False :=
  Iff.rfl


@[simp]
theorem realize_top : (⊤ : L.Formula α).Realize v ↔ True :=
  BoundedFormula.realize_top


@[simp]
theorem realize_inf : (φ ⊓ ψ).Realize v ↔ φ.Realize v ∧ ψ.Realize v :=
  BoundedFormula.realize_inf


@[simp]
theorem realize_imp : (φ.imp ψ).Realize v ↔ φ.Realize v → ψ.Realize v :=
  BoundedFormula.realize_imp


@[simp]
theorem realize_rel {k : ℕ} {R : L.Relations k} {ts : Fin k → L.Term α} :
    (R.formula ts).Realize v ↔ RelMap R fun i => (ts i).realize v :=
                                       /-
                                         L : FirstOrder.Language
                                         M : Type w
                                         inst✝ : L.Structure M
                                         α : Type u'
                                         v : α → M
                                         k : Nat
                                         R : L.Relations k
                                         ts : Fin k → L.Term α
                                         ⊢ Iff (FirstOrder.Language.Structure.RelMap R fun i => FirstOrder.Language.Ter …
                                       -/
  BoundedFormula.realize_rel.trans (by simp)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem realize_rel₁ {R : L.Relations 1} {t : L.Term _} :
    (R.formula₁ t).Realize v ↔ RelMap R ![t.realize v] := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    v : α → M
    R : L.Relations 1
    t : L.Term α
    ⊢ Iff ((R.formula₁ t).Realize v) (FirstOrder.Language.Structure.RelMap R (Matr …
  -/
  rw [Relations.formula₁, realize_rel, iff_eq_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    v : α → M
    R : L.Relations 1
    t : L.Term α
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R fun i => FirstOrder.Language.Term …
  -/
  refine congr rfl (funext fun _ => ?_)
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    v : α → M
    R : L.Relations 1
    t : L.Term α
    x✝ : Fin 1
    ⊢ Eq (FirstOrder.Language.Term.realize v (Matrix.vecCons t Matrix.vecEmpty x✝) …
  -/
  simp only [Matrix.cons_val_fin_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_rel₂ {R : L.Relations 2} {t₁ t₂ : L.Term _} :
    (R.formula₂ t₁ t₂).Realize v ↔ RelMap R ![t₁.realize v, t₂.realize v] := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    v : α → M
    R : L.Relations 2
    t₁ t₂ : L.Term α
    ⊢ Iff ((R.formula₂ t₁ t₂).Realize v) (FirstOrder.Language.Structure.RelMap R ( …
  -/
  rw [Relations.formula₂, realize_rel, iff_eq_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    v : α → M
    R : L.Relations 2
    t₁ t₂ : L.Term α
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R fun i => FirstOrder.Language.Term …
  -/
  refine congr rfl (funext (Fin.cases ?_ ?_))
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      v : α → M
      R : L.Relations 2
      t₁ t₂ : L.Term α
      ⊢ Eq (FirstOrder.Language.Term.realize v (Matrix.vecCons t₁ (Matrix.vecCons t₂ …
    -/
  · simp only [Matrix.cons_val_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      v : α → M
      R : L.Relations 2
      t₁ t₂ : L.Term α
      ⊢ ∀ (i : Fin 1), Eq (FirstOrder.Language.Term.realize v (Matrix.vecCons t₁ (Ma …
    -/
  · simp only [Matrix.cons_val_succ, Matrix.cons_val_fin_one, forall_const]
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_sup : (φ ⊔ ψ).Realize v ↔ φ.Realize v ∨ ψ.Realize v :=
  BoundedFormula.realize_sup


@[simp]
theorem realize_iff : (φ.iff ψ).Realize v ↔ (φ.Realize v ↔ ψ.Realize v) :=
  BoundedFormula.realize_iff


@[simp]
theorem realize_relabel {φ : L.Formula α} {g : α → β} {v : β → M} :
    (φ.relabel g).Realize v ↔ φ.Realize (v ∘ g) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    φ : L.Formula α
    g : α → β
    v : β → M
    ⊢ Iff ((FirstOrder.Language.Formula.relabel g φ).Realize v) (φ.Realize (Functi …
  -/
  rw [Realize, Realize, relabel, BoundedFormula.realize_relabel, iff_eq_eq, Fin.castAdd_zero]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    β : Type v'
    φ : L.Formula α
    g : α → β
    v : β → M
    ⊢ Eq (FirstOrder.Language.BoundedFormula.Realize φ (Function.comp (Sum.elim v  …
  -/
  exact congr rfl (funext finZeroElim)
  /-
    🎉 no goals
  -/


theorem realize_relabel_sum_inr (φ : L.Formula (Fin n)) {v : Empty → M} {x : Fin n → M} :
    (BoundedFormula.relabel Sum.inr φ).Realize v x ↔ φ.Realize x := by
  rw [BoundedFormula.realize_relabel, Formula.Realize, Sum.elim_comp_inr, Fin.castAdd_zero,
    cast_refl, Function.comp_id,
    Subsingleton.elim (x ∘ (natAdd n : Fin 0 → Fin n)) default]


@[simp]
theorem realize_equal {t₁ t₂ : L.Term α} {x : α → M} :
                                                                /-
                                                                  L : FirstOrder.Language
                                                                  M : Type w
                                                                  inst✝ : L.Structure M
                                                                  α : Type u'
                                                                  t₁ t₂ : L.Term α
                                                                  x : α → M
                                                                  ⊢ Iff ((t₁.equal t₂).Realize x) (Eq (FirstOrder.Language.Term.realize x t₁) (F …
                                                                -/
    (t₁.equal t₂).Realize x ↔ t₁.realize x = t₂.realize x := by simp [Term.equal, Realize]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem realize_graph {f : L.Functions n} {x : Fin n → M} {y : M} :
    (Formula.graph f).Realize (Fin.cons y x : _ → M) ↔ funMap f x = y := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    n : Nat
    f : L.Functions n
    x : Fin n → M
    y : M
    ⊢ Iff ((FirstOrder.Language.Formula.graph f).Realize (Fin.cons y x)) (Eq (Firs …
  -/
  simp only [Formula.graph, Term.realize, realize_equal, Fin.cons_zero, Fin.cons_succ]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    n : Nat
    f : L.Functions n
    x : Fin n → M
    y : M
    ⊢ Iff (Eq y (FirstOrder.Language.Structure.funMap f fun i => x i)) (Eq (FirstO …
  -/
  rw [eq_comm]
  /-
    🎉 no goals
  -/


theorem boundedFormula_realize_eq_realize (φ : L.Formula α) (x : α → M) (y : Fin 0 → M) :
    BoundedFormula.Realize φ x y ↔ φ.Realize x := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    φ : L.Formula α
    x : α → M
    y : Fin 0 → M
    ⊢ Iff (FirstOrder.Language.BoundedFormula.Realize φ x y) (φ.Realize x)
  -/
  rw [Formula.Realize, iff_iff_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    φ : L.Formula α
    x : α → M
    y : Fin 0 → M
    ⊢ Eq (FirstOrder.Language.BoundedFormula.Realize φ x y) (FirstOrder.Language.B …
  -/
  congr
  /-
    case e__xs
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    φ : L.Formula α
    x : α → M
    y : Fin 0 → M
    ⊢ Eq y Inhabited.default
  -/
  ext i; exact Fin.elim0 i
         /-
           🎉 no goals
         -/


@[simp]
theorem LHom.realize_onFormula [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M] (ψ : L.Formula α)
    {v : α → M} : (φ.onFormula ψ).Realize v ↔ ψ.Realize v :=
  φ.realize_onBoundedFormula ψ


@[simp]
theorem LHom.setOf_realize_onFormula [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M]
    (ψ : L.Formula α) : (setOf (φ.onFormula ψ).Realize : Set (α → M)) = setOf ψ.Realize := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : L'.Structure M
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    ψ : L.Formula α
    ⊢ Eq (setOf (φ.onFormula ψ).Realize) (setOf ψ.Realize)
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    L' : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : L'.Structure M
    φ : L.LHom L'
    inst✝ : φ.IsExpansionOn M
    ψ : L.Formula α
    x✝ : α → M
    ⊢ Iff (Membership.mem (setOf (φ.onFormula ψ).Realize) x✝) (Membership.mem (set …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A sentence can be evaluated as true or false in a structure. -/
nonrec def Sentence.Realize (φ : L.Sentence) : Prop :=
  φ.Realize (default : _ → M)

-- input using \|= or \vDash, but not using \models

@[inherit_doc Sentence.Realize]
infixl:51 " ⊨ " => Sentence.Realize


@[simp]
theorem Sentence.realize_not {φ : L.Sentence} : M ⊨ φ.not ↔ ¬M ⊨ φ :=
  Iff.rfl


@[simp]
theorem realize_equivSentence_symm_con [L[[α]].Structure M]
    [(L.lhomWithConstants α).IsExpansionOn M] (φ : L[[α]].Sentence) :
    ((equivSentence.symm φ).Realize fun a => (L.con a : M)) ↔ φ.Realize M := by
  simp only [equivSentence, _root_.Equiv.symm_symm, Equiv.coe_trans, Realize,
    BoundedFormula.realize_relabelEquiv, Function.comp]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    φ : (L.withConstants α).Sentence
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.constantsVarsEquiv φ).Realize (Func …
  -/
  refine _root_.trans ?_ BoundedFormula.realize_constantsVarsEquiv
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    φ : (L.withConstants α).Sentence
    ⊢ Iff ((FirstOrder.Language.BoundedFormula.constantsVarsEquiv φ).Realize (Func …
  -/
  rw [iff_iff_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    φ : (L.withConstants α).Sentence
    ⊢ Eq ((FirstOrder.Language.BoundedFormula.constantsVarsEquiv φ).Realize (Funct …
  -/
  congr with (_ | a)
    /-
      case e__v.h.inl
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      φ : (L.withConstants α).Sentence
      val✝ : α
      ⊢ Eq (Function.comp (fun a => ↑(L.con a)) (⇑(Equiv.sumEmpty α Empty)) (Sum.inl …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case e__v.h.inr
      L : FirstOrder.Language
      M : Type w
      inst✝² : L.Structure M
      α : Type u'
      inst✝¹ : (L.withConstants α).Structure M
      inst✝ : (L.lhomWithConstants α).IsExpansionOn M
      φ : (L.withConstants α).Sentence
      a : Empty
      ⊢ Eq (Function.comp (fun a => ↑(L.con a)) (⇑(Equiv.sumEmpty α Empty)) (Sum.inr …
    -/
  · cases a
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_equivSentence [L[[α]].Structure M] [(L.lhomWithConstants α).IsExpansionOn M]
    (φ : L.Formula α) : (equivSentence φ).Realize M ↔ φ.Realize fun a => (L.con a : M) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝² : L.Structure M
    α : Type u'
    inst✝¹ : (L.withConstants α).Structure M
    inst✝ : (L.lhomWithConstants α).IsExpansionOn M
    φ : L.Formula α
    ⊢ Iff (FirstOrder.Language.Sentence.Realize M (FirstOrder.Language.Formula.equ …
  -/
  rw [← realize_equivSentence_symm_con M (equivSentence φ), _root_.Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem realize_equivSentence_symm (φ : L[[α]].Sentence) (v : α → M) :
    (equivSentence.symm φ).Realize v ↔
      @Sentence.Realize _ M (@Language.withConstantsStructure L M _ α (constantsOn.structure v))
        φ :=
  letI := constantsOn.structure v
  realize_equivSentence_symm_con M φ


@[simp]
theorem LHom.realize_onSentence [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M]
    (ψ : L.Sentence) : M ⊨ φ.onSentence ψ ↔ M ⊨ ψ :=
  φ.realize_onFormula ψ


/-- The complete theory of a structure `M` is the set of all sentences `M` satisfies. -/
def completeTheory : L.Theory :=
  { φ | M ⊨ φ }


/-- Two structures are elementarily equivalent when they satisfy the same sentences. -/
def ElementarilyEquivalent : Prop :=
  L.completeTheory M = L.completeTheory N


@[inherit_doc FirstOrder.Language.ElementarilyEquivalent]
scoped[FirstOrder]
  notation:25 A " ≅[" L "] " B:50 => FirstOrder.Language.ElementarilyEquivalent L A B


@[simp]
theorem mem_completeTheory {φ : Sentence L} : φ ∈ L.completeTheory M ↔ M ⊨ φ :=
  Iff.rfl


theorem elementarilyEquivalent_iff : M ≅[L] N ↔ ∀ φ : L.Sentence, M ⊨ φ ↔ N ⊨ φ := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝¹ : L.Structure M
    inst✝ : L.Structure N
    ⊢ Iff (L.ElementarilyEquivalent M N) (∀ (φ : L.Sentence), Iff (FirstOrder.Lang …
  -/
  simp only [ElementarilyEquivalent, Set.ext_iff, completeTheory, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- A model of a theory is a structure in which every sentence is realized as true. -/
class Theory.Model (T : L.Theory) : Prop where
  realize_of_mem : ∀ φ ∈ T, M ⊨ φ

-- input using \|= or \vDash, but not using \models

@[inherit_doc Theory.Model]
infixl:51 " ⊨ " => Theory.Model


@[simp default-10]
theorem Theory.model_iff : M ⊨ T ↔ ∀ φ ∈ T, M ⊨ φ :=
  ⟨fun h => h.realize_of_mem, fun h => ⟨h⟩⟩


theorem Theory.realize_sentence_of_mem [M ⊨ T] {φ : L.Sentence} (h : φ ∈ T) : M ⊨ φ :=
  Theory.Model.realize_of_mem φ h


@[simp]
theorem LHom.onTheory_model [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M] (T : L.Theory) :
                                   /-
                                     L : FirstOrder.Language
                                     L' : FirstOrder.Language
                                     M : Type w
                                     inst✝² : L.Structure M
                                     inst✝¹ : L'.Structure M
                                     φ : L.LHom L'
                                     inst✝ : φ.IsExpansionOn M
                                     T : L.Theory
                                     ⊢ Iff (FirstOrder.Language.Theory.Model M (φ.onTheory T)) (FirstOrder.Language …
                                   -/
    M ⊨ φ.onTheory T ↔ M ⊨ T := by simp [Theory.model_iff, LHom.onTheory]
                                   /-
                                     🎉 no goals
                                   -/


instance model_empty : M ⊨ (∅ : L.Theory) :=
  ⟨fun φ hφ => (Set.not_mem_empty φ hφ).elim⟩


theorem Model.mono {T' : L.Theory} (_h : M ⊨ T') (hs : T ⊆ T') : M ⊨ T :=
  ⟨fun _φ hφ => T'.realize_sentence_of_mem (hs hφ)⟩


theorem Model.union {T' : L.Theory} (h : M ⊨ T) (h' : M ⊨ T') : M ⊨ T ∪ T' := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    T T' : L.Theory
    h : FirstOrder.Language.Theory.Model M T
    h' : FirstOrder.Language.Theory.Model M T'
    ⊢ FirstOrder.Language.Theory.Model M (Union.union T T')
  -/
  simp only [model_iff, Set.mem_union] at *
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    T T' : L.Theory
    h : ∀ (φ : L.Sentence), Membership.mem T φ → FirstOrder.Language.Sentence.Real …
    h' : ∀ (φ : L.Sentence), Membership.mem T' φ → FirstOrder.Language.Sentence.Re …
    ⊢ ∀ (φ : L.Sentence), Or (Membership.mem T φ) (Membership.mem T' φ) → FirstOrd …
  -/
  exact fun φ hφ => hφ.elim (h _) (h' _)
  /-
    🎉 no goals
  -/


@[simp]
theorem model_union_iff {T' : L.Theory} : M ⊨ T ∪ T' ↔ M ⊨ T ∧ M ⊨ T' :=
  ⟨fun h => ⟨h.mono Set.subset_union_left, h.mono Set.subset_union_right⟩, fun h =>
    h.1.union h.2⟩


@[simp]
                                                                                  /-
                                                                                    L : FirstOrder.Language
                                                                                    M : Type w
                                                                                    inst✝ : L.Structure M
                                                                                    φ : L.Sentence
                                                                                    ⊢ Iff (FirstOrder.Language.Theory.Model M (Singleton.singleton φ)) (FirstOrder …
                                                                                  -/
theorem model_singleton_iff {φ : L.Sentence} : M ⊨ ({φ} : L.Theory) ↔ M ⊨ φ := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem model_insert_iff {φ : L.Sentence} : M ⊨ insert φ T ↔ M ⊨ φ ∧ M ⊨ T := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    T : L.Theory
    φ : L.Sentence
    ⊢ Iff (FirstOrder.Language.Theory.Model M (Insert.insert φ T)) (And (FirstOrde …
  -/
  rw [Set.insert_eq, model_union_iff, model_singleton_iff]
  /-
    🎉 no goals
  -/


theorem model_iff_subset_completeTheory : M ⊨ T ↔ T ⊆ L.completeTheory M :=
  T.model_iff


theorem completeTheory.subset [MT : M ⊨ T] : T ⊆ L.completeTheory M :=
  model_iff_subset_completeTheory.1 MT


instance model_completeTheory : M ⊨ L.completeTheory M :=
  Theory.model_iff_subset_completeTheory.2 (subset_refl _)


theorem realize_iff_of_model_completeTheory [N ⊨ L.completeTheory M] (φ : L.Sentence) :
    N ⊨ φ ↔ M ⊨ φ := by
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : FirstOrder.Language.Theory.Model N (L.completeTheory M)
    φ : L.Sentence
    ⊢ Iff (FirstOrder.Language.Sentence.Realize N φ) (FirstOrder.Language.Sentence …
  -/
  refine ⟨fun h => ?_, (L.completeTheory M).realize_sentence_of_mem⟩
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : FirstOrder.Language.Theory.Model N (L.completeTheory M)
    φ : L.Sentence
    h : FirstOrder.Language.Sentence.Realize N φ
    ⊢ FirstOrder.Language.Sentence.Realize M φ
  -/
  contrapose! h
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : FirstOrder.Language.Theory.Model N (L.completeTheory M)
    φ : L.Sentence
    h : Not (FirstOrder.Language.Sentence.Realize M φ)
    ⊢ Not (FirstOrder.Language.Sentence.Realize N φ)
  -/
  rw [← Sentence.realize_not] at *
  /-
    L : FirstOrder.Language
    M : Type w
    N : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : L.Structure N
    inst✝ : FirstOrder.Language.Theory.Model N (L.completeTheory M)
    φ : L.Sentence
    h : FirstOrder.Language.Sentence.Realize M (FirstOrder.Language.Formula.not φ)
    ⊢ FirstOrder.Language.Sentence.Realize N (FirstOrder.Language.Formula.not φ)
  -/
  exact (L.completeTheory M).realize_sentence_of_mem (mem_completeTheory.2 h)
  /-
    🎉 no goals
  -/


@[simp]
theorem realize_alls {φ : L.BoundedFormula α n} {v : α → M} :
    φ.alls.Realize v ↔ ∀ xs : Fin n → M, φ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    ⊢ Iff (φ.alls.Realize v) (∀ (xs : Fin n → M), φ.Realize v xs)
  -/
  induction' n with n ih
    /-
      case zero
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n : Nat
      v : α → M
      φ : L.BoundedFormula α 0
      ⊢ Iff (φ.alls.Realize v) (∀ (xs : Fin 0 → M), φ.Realize v xs)
    -/
  · exact Unique.forall_iff.symm
    /-
      🎉 no goals
    -/
    /-
      case succ
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n✝ : Nat
      v : α → M
      n : Nat
      ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.alls.Realize v) (∀ (xs : Fin n → M), …
      φ : L.BoundedFormula α (HAdd.hAdd n 1)
      ⊢ Iff (φ.alls.Realize v) (∀ (xs : Fin (HAdd.hAdd n 1) → M), φ.Realize v xs)
    -/
  · simp only [alls, ih, Realize]
    /-
      case succ
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n✝ : Nat
      v : α → M
      n : Nat
      ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.alls.Realize v) (∀ (xs : Fin n → M), …
      φ : L.BoundedFormula α (HAdd.hAdd n 1)
      ⊢ Iff (∀ (xs : Fin n → M) (x : M), φ.Realize v (Fin.snoc xs x)) (∀ (xs : Fin ( …
    -/
    exact ⟨fun h xs => Fin.snoc_init_self xs ▸ h _ _, fun h xs x => h (Fin.snoc xs x)⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_exs {φ : L.BoundedFormula α n} {v : α → M} :
    φ.exs.Realize v ↔ ∃ xs : Fin n → M, φ.Realize v xs := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    n : Nat
    φ : L.BoundedFormula α n
    v : α → M
    ⊢ Iff (φ.exs.Realize v) (Exists fun xs => φ.Realize v xs)
  -/
  induction' n with n ih
    /-
      case zero
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n : Nat
      v : α → M
      φ : L.BoundedFormula α 0
      ⊢ Iff (φ.exs.Realize v) (Exists fun xs => φ.Realize v xs)
    -/
  · exact Unique.exists_iff.symm
    /-
      🎉 no goals
    -/
    /-
      case succ
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n✝ : Nat
      v : α → M
      n : Nat
      ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
      φ : L.BoundedFormula α (HAdd.hAdd n 1)
      ⊢ Iff (φ.exs.Realize v) (Exists fun xs => φ.Realize v xs)
    -/
  · simp only [BoundedFormula.exs, ih, realize_ex]
    /-
      case succ
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type u'
      n✝ : Nat
      v : α → M
      n : Nat
      ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
      φ : L.BoundedFormula α (HAdd.hAdd n 1)
      ⊢ Iff (Exists fun xs => Exists fun a => φ.Realize v (Fin.snoc xs a)) (Exists f …
    -/
    constructor
      /-
        case succ.mp
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        n✝ : Nat
        v : α → M
        n : Nat
        ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
        φ : L.BoundedFormula α (HAdd.hAdd n 1)
        ⊢ (Exists fun xs => Exists fun a => φ.Realize v (Fin.snoc xs a)) → Exists fun  …
      -/
    · rintro ⟨xs, x, h⟩
      /-
        case succ.mp.intro.intro
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        n✝ : Nat
        v : α → M
        n : Nat
        ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
        φ : L.BoundedFormula α (HAdd.hAdd n 1)
        xs : Fin n → M
        x : M
        h : φ.Realize v (Fin.snoc xs x)
        ⊢ Exists fun xs => φ.Realize v xs
      -/
      exact ⟨_, h⟩
      /-
        🎉 no goals
      -/
      /-
        case succ.mpr
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        n✝ : Nat
        v : α → M
        n : Nat
        ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
        φ : L.BoundedFormula α (HAdd.hAdd n 1)
        ⊢ (Exists fun xs => φ.Realize v xs) → Exists fun xs => Exists fun a => φ.Reali …
      -/
    · rintro ⟨xs, h⟩
      /-
        case succ.mpr.intro
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        n✝ : Nat
        v : α → M
        n : Nat
        ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
        φ : L.BoundedFormula α (HAdd.hAdd n 1)
        xs : Fin (HAdd.hAdd n 1) → M
        h : φ.Realize v xs
        ⊢ Exists fun xs => Exists fun a => φ.Realize v (Fin.snoc xs a)
      -/
      rw [← Fin.snoc_init_self xs] at h
      /-
        case succ.mpr.intro
        L : FirstOrder.Language
        M : Type w
        inst✝ : L.Structure M
        α : Type u'
        n✝ : Nat
        v : α → M
        n : Nat
        ih : ∀ {φ : L.BoundedFormula α n}, Iff (φ.exs.Realize v) (Exists fun xs => φ.R …
        φ : L.BoundedFormula α (HAdd.hAdd n 1)
        xs : Fin (HAdd.hAdd n 1) → M
        h : φ.Realize v (Fin.snoc (Fin.init xs) (xs (Fin.last n)))
        ⊢ Exists fun xs => Exists fun a => φ.Realize v (Fin.snoc xs a)
      -/
      exact ⟨_, _, h⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem _root_.FirstOrder.Language.Formula.realize_iAlls
    [Finite β] {φ : L.Formula (α ⊕ β)} {v : α → M} : (φ.iAlls β).Realize v ↔
      ∀ (i : β → M), φ.Realize (fun a => Sum.elim v i a) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    β : Type v'
    inst✝ : Finite β
    φ : L.Formula (Sum α β)
    v : α → M
    ⊢ Iff ((FirstOrder.Language.Formula.iAlls β φ).Realize v) (∀ (i : β → M), φ.Re …
  -/
  let e := Classical.choice (Classical.choose_spec (Finite.exists_equiv_fin β))
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    β : Type v'
    inst✝ : Finite β
    φ : L.Formula (Sum α β)
    v : α → M
    e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
    ⊢ Iff ((FirstOrder.Language.Formula.iAlls β φ).Realize v) (∀ (i : β → M), φ.Re …
  -/
  rw [Formula.iAlls]
  simp only [Nat.add_zero, realize_alls, realize_relabel, Function.comp_def,
    castAdd_zero, finCongr_refl, OrderIso.refl_apply, Sum.elim_map, id_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    β : Type v'
    inst✝ : Finite β
    φ : L.Formula (Sum α β)
    v : α → M
    e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
    ⊢ Iff (∀ (xs : Fin (Classical.choose ⋯) → M), FirstOrder.Language.BoundedFormu …
  -/
  refine Equiv.forall_congr ?_ ?_
  · exact ⟨fun v => v ∘ e, fun v => v ∘ e.symm,
      fun _ => by simp [Function.comp_def],
      fun _ => by simp [Function.comp_def]⟩
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      β : Type v'
      inst✝ : Finite β
      φ : L.Formula (Sum α β)
      v : α → M
      e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      ⊢ ∀ (a : Fin (Classical.choose ⋯) → M), Iff (FirstOrder.Language.BoundedFormul …
    -/
  · intro x
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      β : Type v'
      inst✝ : Finite β
      φ : L.Formula (Sum α β)
      v : α → M
      e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      ⊢ Iff (FirstOrder.Language.BoundedFormula.Realize φ (fun x_1 => Sum.elim (fun  …
    -/
    rw [Formula.Realize, iff_iff_eq]
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      β : Type v'
      inst✝ : Finite β
      φ : L.Formula (Sum α β)
      v : α → M
      e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      ⊢ Eq (FirstOrder.Language.BoundedFormula.Realize φ (fun x_1 => Sum.elim (fun x …
    -/
    congr
    /-
      case refine_2.e__xs
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      β : Type v'
      inst✝ : Finite β
      φ : L.Formula (Sum α β)
      v : α → M
      e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      ⊢ Eq (fun x_1 => x (Fin.natAdd (Classical.choose ⋯) x_1)) Inhabited.default
    -/
    funext i
    /-
      case refine_2.e__xs.h
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      β : Type v'
      inst✝ : Finite β
      φ : L.Formula (Sum α β)
      v : α → M
      e : _root_.Equiv β (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      i : Fin 0
      ⊢ Eq (x (Fin.natAdd (Classical.choose ⋯) i)) (Inhabited.default i)
    -/
    exact i.elim0
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_iAlls [Finite β] {φ : L.Formula (α ⊕ β)} {v : α → M} {v' : Fin 0 → M} :
    BoundedFormula.Realize (φ.iAlls β) v v' ↔
      ∀ (i : β → M), φ.Realize (fun a => Sum.elim v i a) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    β : Type v'
    inst✝ : Finite β
    φ : L.Formula (Sum α β)
    v : α → M
    v' : Fin 0 → M
    ⊢ Iff (FirstOrder.Language.BoundedFormula.Realize (FirstOrder.Language.Formula …
  -/
  rw [← Formula.realize_iAlls, iff_iff_eq]; congr; simp [eq_iff_true_of_subsingleton]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem _root_.FirstOrder.Language.Formula.realize_iExs
    [Finite γ] {φ : L.Formula (α ⊕ γ)} {v : α → M} : (φ.iExs γ).Realize v ↔
      ∃ (i : γ → M), φ.Realize (Sum.elim v i) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    γ : Type u_3
    inst✝ : Finite γ
    φ : L.Formula (Sum α γ)
    v : α → M
    ⊢ Iff ((FirstOrder.Language.Formula.iExs γ φ).Realize v) (Exists fun i => φ.Re …
  -/
  let e := Classical.choice (Classical.choose_spec (Finite.exists_equiv_fin γ))
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    γ : Type u_3
    inst✝ : Finite γ
    φ : L.Formula (Sum α γ)
    v : α → M
    e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
    ⊢ Iff ((FirstOrder.Language.Formula.iExs γ φ).Realize v) (Exists fun i => φ.Re …
  -/
  rw [Formula.iExs]
  simp only [Nat.add_zero, realize_exs, realize_relabel, Function.comp_def,
    castAdd_zero, finCongr_refl, OrderIso.refl_apply, Sum.elim_map, id_eq]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    γ : Type u_3
    inst✝ : Finite γ
    φ : L.Formula (Sum α γ)
    v : α → M
    e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
    ⊢ Iff (Exists fun xs => FirstOrder.Language.BoundedFormula.Realize φ (fun x => …
  -/
  refine Equiv.exists_congr ?_ ?_
  · exact ⟨fun v => v ∘ e, fun v => v ∘ e.symm,
      fun _ => by simp [Function.comp_def],
      fun _ => by simp [Function.comp_def]⟩
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      γ : Type u_3
      inst✝ : Finite γ
      φ : L.Formula (Sum α γ)
      v : α → M
      e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      ⊢ ∀ (a : Fin (Classical.choose ⋯) → M), Iff (FirstOrder.Language.BoundedFormul …
    -/
  · intro x
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      γ : Type u_3
      inst✝ : Finite γ
      φ : L.Formula (Sum α γ)
      v : α → M
      e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      ⊢ Iff (FirstOrder.Language.BoundedFormula.Realize φ (fun x_1 => Sum.elim (fun  …
    -/
    rw [Formula.Realize, iff_iff_eq]
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      γ : Type u_3
      inst✝ : Finite γ
      φ : L.Formula (Sum α γ)
      v : α → M
      e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      ⊢ Eq (FirstOrder.Language.BoundedFormula.Realize φ (fun x_1 => Sum.elim (fun x …
    -/
    congr
    /-
      case refine_2.e__xs
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      γ : Type u_3
      inst✝ : Finite γ
      φ : L.Formula (Sum α γ)
      v : α → M
      e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      ⊢ Eq (fun x_1 => x (Fin.natAdd (Classical.choose ⋯) x_1)) Inhabited.default
    -/
    funext i
    /-
      case refine_2.e__xs.h
      L : FirstOrder.Language
      M : Type w
      inst✝¹ : L.Structure M
      α : Type u'
      γ : Type u_3
      inst✝ : Finite γ
      φ : L.Formula (Sum α γ)
      v : α → M
      e : _root_.Equiv γ (Fin (Classical.choose ⋯)) := Classical.choice ⋯
      x : Fin (Classical.choose ⋯) → M
      i : Fin 0
      ⊢ Eq (x (Fin.natAdd (Classical.choose ⋯) i)) (Inhabited.default i)
    -/
    exact i.elim0
    /-
      🎉 no goals
    -/


@[simp]
theorem realize_iExs [Finite γ] {φ : L.Formula (α ⊕ γ)} {v : α → M} {v' : Fin 0 → M} :
    BoundedFormula.Realize (φ.iExs γ) v v' ↔
      ∃ (i : γ → M), φ.Realize (Sum.elim v i) := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u'
    γ : Type u_3
    inst✝ : Finite γ
    φ : L.Formula (Sum α γ)
    v : α → M
    v' : Fin 0 → M
    ⊢ Iff (FirstOrder.Language.BoundedFormula.Realize (FirstOrder.Language.Formula …
  -/
  rw [← Formula.realize_iExs, iff_iff_eq]; congr; simp [eq_iff_true_of_subsingleton]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem realize_toFormula (φ : L.BoundedFormula α n) (v : α ⊕ (Fin n) → M) :
    φ.toFormula.Realize v ↔ φ.Realize (v ∘ Sum.inl) (v ∘ Sum.inr) := by
  induction φ with
  | falsum => rfl
  | equal => simp [BoundedFormula.Realize]
  | rel => simp [BoundedFormula.Realize]
  | imp _ _ ih1 ih2 =>
    rw [toFormula, Formula.Realize, realize_imp, ← Formula.Realize, ih1, ← Formula.Realize, ih2,
      realize_imp]
  | all _ ih3 =>
    rw [toFormula, Formula.Realize, realize_all, realize_all]
    refine forall_congr' fun a => ?_
    have h := ih3 (Sum.elim (v ∘ Sum.inl) (snoc (v ∘ Sum.inr) a))
    simp only [Sum.elim_comp_inl, Sum.elim_comp_inr] at h
    rw [← h, realize_relabel, Formula.Realize, iff_iff_eq]
    simp only [Function.comp_def]
    congr with x
    · cases' x with _ x
      · simp
      · refine Fin.lastCases ?_ ?_ x
        · rw [Sum.elim_inr, Sum.elim_inr,
            finSumFinEquiv_symm_last, Sum.map_inr, Sum.elim_inr]
          simp [Fin.snoc]
        · simp only [castSucc, Function.comp_apply, Sum.elim_inr,
            finSumFinEquiv_symm_apply_castAdd, Sum.map_inl, Sum.elim_inl]
          rw [← castSucc]
          simp
    · exact Fin.elim0 x


@[simp]
theorem realize_iSup [Finite β] (f : β → L.BoundedFormula α n)
    (v : α → M) (v' : Fin n → M) :
    (iSup f).Realize v v' ↔ ∃ b, (f b).Realize v v' := by
  simp only [iSup, realize_foldr_sup, List.mem_map, Finset.mem_toList, Finset.mem_univ, true_and,
    exists_exists_eq_and]


@[simp]
theorem realize_iInf [Finite β] (f : β → L.BoundedFormula α n)
    (v : α → M) (v' : Fin n → M) :
    (iInf f).Realize v v' ↔ ∀ b, (f b).Realize v v' := by
  simp only [iInf, realize_foldr_inf, List.mem_map, Finset.mem_toList, Finset.mem_univ, true_and,
    forall_exists_index, forall_apply_eq_imp_iff]


@[simp]
theorem realize_boundedFormula (φ : L.BoundedFormula α n) {v : α → M}
    {xs : Fin n → M} : φ.Realize (g ∘ v) (g ∘ xs) ↔ φ.Realize v xs := by
  induction φ with
  | falsum => rfl
  | equal =>
    simp only [BoundedFormula.Realize, ← Sum.comp_elim, HomClass.realize_term,
      EmbeddingLike.apply_eq_iff_eq g]
  | rel =>
    simp only [BoundedFormula.Realize, ← Sum.comp_elim, HomClass.realize_term]
    exact StrongHomClass.map_rel g _ _
  | imp _ _ ih1 ih2 => rw [BoundedFormula.Realize, ih1, ih2, BoundedFormula.Realize]
  | all _ ih3 =>
    rw [BoundedFormula.Realize, BoundedFormula.Realize]
    constructor
    · intro h a
      have h' := h (g a)
      rw [← Fin.comp_snoc, ih3] at h'
      exact h'
    · intro h a
      have h' := h (EquivLike.inv g a)
      rw [← ih3, Fin.comp_snoc, EquivLike.apply_inv_apply g] at h'
      exact h'


@[simp]
theorem realize_formula (φ : L.Formula α) {v : α → M} :
    φ.Realize (g ∘ v) ↔ φ.Realize v := by
  rw [Formula.Realize, Formula.Realize, ← realize_boundedFormula g φ, iff_eq_eq,
    Unique.eq_default (g ∘ default)]


theorem realize_sentence (φ : L.Sentence) : M ⊨ φ ↔ N ⊨ φ := by
  rw [Sentence.Realize, Sentence.Realize, ← realize_formula g,
    Unique.eq_default (g ∘ default)]


theorem theory_model [M ⊨ T] : N ⊨ T :=
  ⟨fun φ hφ => (realize_sentence g φ).1 (Theory.realize_sentence_of_mem T hφ)⟩


theorem elementarilyEquivalent : M ≅[L] N :=
  elementarilyEquivalent_iff.2 (realize_sentence g)


@[simp]
theorem realize_reflexive : M ⊨ r.reflexive ↔ Reflexive fun x y : M => RelMap r ![x, y] :=
  forall_congr' fun _ => realize_rel₂


@[simp]
theorem realize_irreflexive : M ⊨ r.irreflexive ↔ Irreflexive fun x y : M => RelMap r ![x, y] :=
  forall_congr' fun _ => not_congr realize_rel₂


@[simp]
theorem realize_symmetric : M ⊨ r.symmetric ↔ Symmetric fun x y : M => RelMap r ![x, y] :=
  forall_congr' fun _ => forall_congr' fun _ => imp_congr realize_rel₂ realize_rel₂


@[simp]
theorem realize_antisymmetric :
    M ⊨ r.antisymmetric ↔ AntiSymmetric fun x y : M => RelMap r ![x, y] :=
  forall_congr' fun _ =>
    forall_congr' fun _ => imp_congr realize_rel₂ (imp_congr realize_rel₂ Iff.rfl)


@[simp]
theorem realize_transitive : M ⊨ r.transitive ↔ Transitive fun x y : M => RelMap r ![x, y] :=
  forall_congr' fun _ =>
    forall_congr' fun _ =>
      forall_congr' fun _ => imp_congr realize_rel₂ (imp_congr realize_rel₂ realize_rel₂)


@[simp]
theorem realize_total : M ⊨ r.total ↔ Total fun x y : M => RelMap r ![x, y] :=
  forall_congr' fun _ =>
    forall_congr' fun _ => realize_sup.trans (or_congr realize_rel₂ realize_rel₂)


@[simp]
theorem Sentence.realize_cardGe (n) : M ⊨ Sentence.cardGe L n ↔ ↑n ≤ #M := by
  rw [← lift_mk_fin, ← lift_le.{0}, lift_lift, lift_mk_le, Sentence.cardGe, Sentence.Realize,
    BoundedFormula.realize_exs]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    n : Nat
    ⊢ Iff (Exists fun xs => (List.foldr (fun x1 x2 => Min.min x1 x2) Top.top (List …
  -/
  simp_rw [BoundedFormula.realize_foldr_inf]
  simp only [Function.comp_apply, List.mem_map, Prod.exists, Ne, List.mem_product,
    List.mem_finRange, forall_exists_index, and_imp, List.mem_filter, true_and]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    n : Nat
    ⊢ Iff (Exists fun xs => ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq ( …
  -/
  refine ⟨?_, fun xs => ⟨xs.some, ?_⟩⟩
    /-
      case refine_1
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      ⊢ (Exists fun xs => ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq (Deci …
    -/
  · rintro ⟨xs, h⟩
    /-
      case refine_1.intro
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      xs : Fin n → M
      h : ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq (Decidable.decide (No …
      ⊢ Nonempty (Function.Embedding (Fin n) M)
    -/
    refine ⟨⟨xs, fun i j ij => ?_⟩⟩
    /-
      case refine_1.intro
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      xs : Fin n → M
      h : ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq (Decidable.decide (No …
      i j : Fin n
      ij : Eq (xs i) (xs j)
      ⊢ Eq i j
    -/
    contrapose! ij
    /-
      case refine_1.intro
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      xs : Fin n → M
      h : ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq (Decidable.decide (No …
      i j : Fin n
      ij : Ne i j
      ⊢ Ne (xs i) (xs j)
    -/
    have hij := h _ i j (by simpa using ij) rfl
    simp only [BoundedFormula.realize_not, Term.realize, BoundedFormula.realize_bdEqual,
      Sum.elim_inr] at hij
    /-
      case refine_1.intro
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      xs : Fin n → M
      h : ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq (Decidable.decide (No …
      i j : Fin n
      ij : Ne i j
      hij : Not (Eq (xs i) (xs j))
      ⊢ Ne (xs i) (xs j)
    -/
    exact hij
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      xs : Nonempty (Function.Embedding (Fin n) M)
      ⊢ ∀ (φ : L.BoundedFormula Empty n) (x x_1 : Fin n), Eq (Decidable.decide (Not  …
    -/
  · rintro _ i j ij rfl
    /-
      case refine_2
      L : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      n : Nat
      xs : Nonempty (Function.Embedding (Fin n) M)
      i j : Fin n
      ij : Eq (Decidable.decide (Not (Eq i j))) Bool.true
      ⊢ ((FirstOrder.Language.Term.var (Sum.inr i)).bdEqual (FirstOrder.Language.Ter …
    -/
    simpa using ij
    /-
      🎉 no goals
    -/


@[simp]
theorem model_infiniteTheory_iff : M ⊨ L.infiniteTheory ↔ Infinite M := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    ⊢ Iff (FirstOrder.Language.Theory.Model M L.infiniteTheory) (Infinite M)
  -/
  simp [infiniteTheory, infinite_iff, aleph0_le]
  /-
    🎉 no goals
  -/


instance model_infiniteTheory [h : Infinite M] : M ⊨ L.infiniteTheory :=
  L.model_infiniteTheory_iff.2 h


@[simp]
theorem model_nonemptyTheory_iff : M ⊨ L.nonemptyTheory ↔ Nonempty M := by
  simp only [nonemptyTheory, Theory.model_iff, Set.mem_singleton_iff, forall_eq,
    Sentence.realize_cardGe, Nat.cast_one, one_le_iff_ne_zero, mk_ne_zero_iff]


instance model_nonempty [h : Nonempty M] : M ⊨ L.nonemptyTheory :=
  L.model_nonemptyTheory_iff.2 h


theorem model_distinctConstantsTheory {M : Type w} [L[[α]].Structure M] (s : Set α) :
    M ⊨ L.distinctConstantsTheory s ↔ Set.InjOn (fun i : α => (L.con i : M)) s := by
  simp only [distinctConstantsTheory, Theory.model_iff, Set.mem_image, Set.mem_inter,
    Set.mem_prod, Set.mem_compl, Prod.exists, forall_exists_index, and_imp]
  /-
    L : FirstOrder.Language
    α : Type u'
    M : Type w
    inst✝ : (L.withConstants α).Structure M
    s : Set α
    ⊢ Iff (∀ (φ : (L.withConstants α).Sentence) (x x_1 : α), Membership.mem (Inter …
  -/
  refine ⟨fun h a as b bs ab => ?_, ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      α : Type u'
      M : Type w
      inst✝ : (L.withConstants α).Structure M
      s : Set α
      h : ∀ (φ : (L.withConstants α).Sentence) (x x_1 : α), Membership.mem (Inter.in …
      a : α
      as : Membership.mem s a
      b : α
      bs : Membership.mem s b
      ab : Eq ((fun i => ↑(L.con i)) a) ((fun i => ↑(L.con i)) b)
      ⊢ Eq a b
    -/
  · contrapose! ab
    /-
      case refine_1
      L : FirstOrder.Language
      α : Type u'
      M : Type w
      inst✝ : (L.withConstants α).Structure M
      s : Set α
      h : ∀ (φ : (L.withConstants α).Sentence) (x x_1 : α), Membership.mem (Inter.in …
      a : α
      as : Membership.mem s a
      b : α
      bs : Membership.mem s b
      ab : Ne a b
      ⊢ Ne ↑(L.con a) ↑(L.con b)
    -/
    have h' := h _ a b ⟨⟨as, bs⟩, ab⟩ rfl
    simp only [Sentence.Realize, Formula.realize_not, Formula.realize_equal,
      Term.realize_constants] at h'
    /-
      case refine_1
      L : FirstOrder.Language
      α : Type u'
      M : Type w
      inst✝ : (L.withConstants α).Structure M
      s : Set α
      h : ∀ (φ : (L.withConstants α).Sentence) (x x_1 : α), Membership.mem (Inter.in …
      a : α
      as : Membership.mem s a
      b : α
      bs : Membership.mem s b
      ab : Ne a b
      h' : Not (Eq ↑(L.con a) ↑(L.con b))
      ⊢ Ne ↑(L.con a) ↑(L.con b)
    -/
    exact h'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      α : Type u'
      M : Type w
      inst✝ : (L.withConstants α).Structure M
      s : Set α
      ⊢ Set.InjOn (fun i => ↑(L.con i)) s → ∀ (φ : (L.withConstants α).Sentence) (x  …
    -/
  · rintro h φ a b ⟨⟨as, bs⟩, ab⟩ rfl
    /-
      case refine_2.intro.intro
      L : FirstOrder.Language
      α : Type u'
      M : Type w
      inst✝ : (L.withConstants α).Structure M
      s : Set α
      h : Set.InjOn (fun i => ↑(L.con i)) s
      a b : α
      ab : Membership.mem (HasCompl.compl (Set.diagonal α)) { fst := a, snd := b }
      as : Membership.mem s { fst := a, snd := b }.1
      bs : Membership.mem s { fst := a, snd := b }.2
      ⊢ FirstOrder.Language.Sentence.Realize M ((L.con a).term.equal (L.con b).term) …
    -/
    simp only [Sentence.Realize, Formula.realize_not, Formula.realize_equal, Term.realize_constants]
    /-
      case refine_2.intro.intro
      L : FirstOrder.Language
      α : Type u'
      M : Type w
      inst✝ : (L.withConstants α).Structure M
      s : Set α
      h : Set.InjOn (fun i => ↑(L.con i)) s
      a b : α
      ab : Membership.mem (HasCompl.compl (Set.diagonal α)) { fst := a, snd := b }
      as : Membership.mem s { fst := a, snd := b }.1
      bs : Membership.mem s { fst := a, snd := b }.2
      ⊢ Not (Eq ↑(L.con a) ↑(L.con b))
    -/
    exact fun contra => ab (h as bs contra)
    /-
      🎉 no goals
    -/


theorem card_le_of_model_distinctConstantsTheory (s : Set α) (M : Type w) [L[[α]].Structure M]
    [h : M ⊨ L.distinctConstantsTheory s] : Cardinal.lift.{w} #s ≤ Cardinal.lift.{u'} #M :=
  lift_mk_le'.2 ⟨⟨_, Set.injOn_iff_injective.1 ((L.model_distinctConstantsTheory s).1 h)⟩⟩


@[symm]
nonrec theorem symm (h : M ≅[L] N) : N ≅[L] M :=
  h.symm


@[trans]
nonrec theorem trans (MN : M ≅[L] N) (NP : N ≅[L] P) : M ≅[L] P :=
  MN.trans NP


theorem completeTheory_eq (h : M ≅[L] N) : L.completeTheory M = L.completeTheory N :=
  h


theorem realize_sentence (h : M ≅[L] N) (φ : L.Sentence) : M ⊨ φ ↔ N ⊨ φ :=
  (elementarilyEquivalent_iff.1 h) φ


theorem theory_model_iff (h : M ≅[L] N) : M ⊨ T ↔ N ⊨ T := by
  rw [Theory.model_iff_subset_completeTheory, Theory.model_iff_subset_completeTheory,
    h.completeTheory_eq]


theorem theory_model [MT : M ⊨ T] (h : M ≅[L] N) : N ⊨ T :=
  h.theory_model_iff.1 MT


theorem nonempty_iff (h : M ≅[L] N) : Nonempty M ↔ Nonempty N :=
  (model_nonemptyTheory_iff L).symm.trans (h.theory_model_iff.trans (model_nonemptyTheory_iff L))


theorem nonempty [Mn : Nonempty M] (h : M ≅[L] N) : Nonempty N :=
  h.nonempty_iff.1 Mn


theorem infinite_iff (h : M ≅[L] N) : Infinite M ↔ Infinite N :=
  (model_infiniteTheory_iff L).symm.trans (h.theory_model_iff.trans (model_infiniteTheory_iff L))


theorem infinite [Mi : Infinite M] (h : M ≅[L] N) : Infinite N :=
  h.infinite_iff.1 Mi


