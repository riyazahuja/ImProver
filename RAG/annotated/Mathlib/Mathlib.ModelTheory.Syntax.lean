/-- A term on `α` is either a variable indexed by an element of `α`
  or a function symbol applied to simpler terms. -/
inductive Term (α : Type u') : Type max u u'
  | var : α → Term α
  | func : ∀ {l : ℕ} (_f : L.Functions l) (_ts : Fin l → Term α), Term α

instance instDecidableEq [DecidableEq α] [∀ n, DecidableEq (L.Functions n)] : DecidableEq (L.Term α)
                                                     /-
                                                       L : FirstOrder.Language
                                                       L' : FirstOrder.Language
                                                       M : Type w
                                                       α : Type u'
                                                       β : Type v'
                                                       γ : Type u_1
                                                       inst✝¹ : DecidableEq α
                                                       inst✝ : (n : Nat) → DecidableEq (L.Functions n)
                                                       a b : α
                                                       ⊢ Iff (Eq a b) (Eq (FirstOrder.Language.Term.var a) (FirstOrder.Language.Term. …
                                                     -/
  | .var a, .var b => decidable_of_iff (a = b) <| by simp
                                                     /-
                                                       🎉 no goals
                                                     -/
  | @Term.func _ _ m f xs, @Term.func _ _ n g ys =>
      if h : m = n then
        letI : DecidableEq (L.Term α) := instDecidableEq
        decidable_of_iff (f = h ▸ g ∧ ∀ i : Fin m, xs i = ys (Fin.cast h i)) <| by
          /-
            L : FirstOrder.Language
            L' : FirstOrder.Language
            M : Type w
            α : Type u'
            β : Type v'
            γ : Type u_1
            inst✝¹ : DecidableEq α
            inst✝ : (n : Nat) → DecidableEq (L.Functions n)
            m : Nat
            f : L.Functions m
            xs : Fin m → L.Term α
            n : Nat
            g : L.Functions n
            ys : Fin n → L.Term α
            h : Eq m n
            this : DecidableEq (L.Term α) := instDecidableEq
            ⊢ Iff (And (Eq f (Eq.rec g ⋯)) (∀ (i : Fin m), Eq (xs i) (ys (Fin.cast h i)))) …
          -/
          subst h
          /-
            L : FirstOrder.Language
            L' : FirstOrder.Language
            M : Type w
            α : Type u'
            β : Type v'
            γ : Type u_1
            inst✝¹ : DecidableEq α
            inst✝ : (n : Nat) → DecidableEq (L.Functions n)
            m : Nat
            f : L.Functions m
            xs : Fin m → L.Term α
            this : DecidableEq (L.Term α) := instDecidableEq
            g : L.Functions m
            ys : Fin m → L.Term α
            ⊢ Iff (And (Eq f (Eq.rec g ⋯)) (∀ (i : Fin m), Eq (xs i) (ys (Fin.cast ⋯ i)))) …
          -/
          simp [funext_iff]
          /-
            🎉 no goals
          -/
      else
                       /-
                         L : FirstOrder.Language
                         L' : FirstOrder.Language
                         M : Type w
                         α : Type u'
                         β : Type v'
                         γ : Type u_1
                         inst✝¹ : DecidableEq α
                         inst✝ : (n : Nat) → DecidableEq (L.Functions n)
                         m : Nat
                         f : L.Functions m
                         xs : Fin m → L.Term α
                         n : Nat
                         g : L.Functions n
                         ys : Fin n → L.Term α
                         h : Not (Eq m n)
                         ⊢ Not (Eq (FirstOrder.Language.Term.func f xs) (FirstOrder.Language.Term.func  …
                       -/
        .isFalse <| by simp [h]
                       /-
                         🎉 no goals
                       -/
                                                            /-
                                                              L : FirstOrder.Language
                                                              L' : FirstOrder.Language
                                                              M : Type w
                                                              α : Type u'
                                                              β : Type v'
                                                              γ : Type u_1
                                                              inst✝¹ : DecidableEq α
                                                              inst✝ : (n : Nat) → DecidableEq (L.Functions n)
                                                              a✝ : α
                                                              l✝ : Nat
                                                              _f✝ : L.Functions l✝
                                                              _ts✝ : Fin l✝ → L.Term α
                                                              ⊢ Not (Eq (FirstOrder.Language.Term.var a✝) (FirstOrder.Language.Term.func _f✝ …
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  | .var _, .func _ _ | .func _ _, .var _ => .isFalse <| by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The `Finset` of variables used in a given term. -/
@[simp]
def varFinset [DecidableEq α] : L.Term α → Finset α
  | var i => {i}
  | func _f ts => univ.biUnion fun i => (ts i).varFinset

-- Porting note: universes in different order

/-- The `Finset` of variables from the left side of a sum used in a given term. -/
@[simp]
def varFinsetLeft [DecidableEq α] : L.Term (α ⊕ β) → Finset α
  | var (Sum.inl i) => {i}
  | var (Sum.inr _i) => ∅
  | func _f ts => univ.biUnion fun i => (ts i).varFinsetLeft

-- Porting note: universes in different order

@[simp]
def relabel (g : α → β) : L.Term α → L.Term β
  | var i => var (g i)
  | func f ts => func f fun {i} => (ts i).relabel g


theorem relabel_id (t : L.Term α) : t.relabel id = t := by
  induction t with
  | var => rfl
  | func _ _ ih => simp [ih]


@[simp]
theorem relabel_id_eq_id : (Term.relabel id : L.Term α → L.Term α) = id :=
  funext relabel_id


@[simp]
theorem relabel_relabel (f : α → β) (g : β → γ) (t : L.Term α) :
    (t.relabel f).relabel g = t.relabel (g ∘ f) := by
  induction t with
  | var => rfl
  | func _ _ ih => simp [ih]


@[simp]
theorem relabel_comp_relabel (f : α → β) (g : β → γ) :
    (Term.relabel g ∘ Term.relabel f : L.Term α → L.Term γ) = Term.relabel (g ∘ f) :=
  funext (relabel_relabel f g)


/-- Relabels a term's variables along a bijection. -/
@[simps]
def relabelEquiv (g : α ≃ β) : L.Term α ≃ L.Term β :=
                                          /-
                                            L : FirstOrder.Language
                                            L' : FirstOrder.Language
                                            M : Type w
                                            α : Type u'
                                            β : Type v'
                                            γ : Type u_1
                                            g : _root_.Equiv α β
                                            t : L.Term α
                                            ⊢ Eq (FirstOrder.Language.Term.relabel (⇑g.symm) (FirstOrder.Language.Term.rel …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  ⟨relabel g, relabel g.symm, fun t => by simp, fun t => by simp⟩
                                                            /-
                                                              🎉 no goals
                                                            -/

-- Porting note: universes in different order

/-- Restricts a term to use only a set of the given variables. -/
def restrictVar [DecidableEq α] : ∀ (t : L.Term α) (_f : t.varFinset → β), L.Term β
  | var a, f => var (f ⟨a, mem_singleton_self a⟩)
  | func F ts, f =>
    func F fun i => (ts i).restrictVar (f ∘ Set.inclusion
      (subset_biUnion_of_mem (fun i => varFinset (ts i)) (mem_univ i)))

-- Porting note: universes in different order

/-- Restricts a term to use only a set of the given variables on the left side of a sum. -/
def restrictVarLeft [DecidableEq α] {γ : Type*} :
    ∀ (t : L.Term (α ⊕ γ)) (_f : t.varFinsetLeft → β), L.Term (β ⊕ γ)
  | var (Sum.inl a), f => var (Sum.inl (f ⟨a, mem_singleton_self a⟩))
  | var (Sum.inr a), _f => var (Sum.inr a)
  | func F ts, f =>
    func F fun i =>
      (ts i).restrictVarLeft (f ∘ Set.inclusion (subset_biUnion_of_mem
        (fun i => varFinsetLeft (ts i)) (mem_univ i)))


/-- The representation of a constant symbol as a term. -/
def Constants.term (c : L.Constants) : L.Term α :=
  func c default


/-- Applies a unary function to a term. -/
def Functions.apply₁ (f : L.Functions 1) (t : L.Term α) : L.Term α :=
  func f ![t]


/-- Applies a binary function to two terms. -/
def Functions.apply₂ (f : L.Functions 2) (t₁ t₂ : L.Term α) : L.Term α :=
  func f ![t₁, t₂]


/-- Sends a term with constants to a term with extra variables. -/
@[simp]
def constantsToVars : L[[γ]].Term α → L.Term (γ ⊕ α)
  | var a => var (Sum.inr a)
  | @func _ _ 0 f ts =>
    Sum.casesOn f (fun f => func f fun i => (ts i).constantsToVars) fun c => var (Sum.inl c)
  | @func _ _ (_n + 1) f ts =>
    Sum.casesOn f (fun f => func f fun i => (ts i).constantsToVars) fun c => isEmptyElim c

-- Porting note: universes in different order

/-- Sends a term with extra variables to a term with constants. -/
@[simp]
def varsToConstants : L.Term (γ ⊕ α) → L[[γ]].Term α
  | var (Sum.inr a) => var a
  | var (Sum.inl c) => Constants.term (Sum.inr c)
  | func f ts => func (Sum.inl f) fun i => (ts i).varsToConstants


/-- A bijection between terms with constants and terms with extra variables. -/
@[simps]
def constantsVarsEquiv : L[[γ]].Term α ≃ L.Term (γ ⊕ α) :=
  ⟨constantsToVars, varsToConstants, by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      α : Type u'
      β : Type v'
      γ : Type u_1
      ⊢ Function.LeftInverse FirstOrder.Language.Term.varsToConstants FirstOrder.Lan …
    -/
    intro t
    induction t with
    | var => rfl
    | @func n f _ ih =>
      cases n
      · cases f
        · simp [constantsToVars, varsToConstants, ih]
        · simp [constantsToVars, varsToConstants, Constants.term, eq_iff_true_of_subsingleton]
      · cases' f with f f
        · simp [constantsToVars, varsToConstants, ih]
        · exact isEmptyElim f, by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      α : Type u'
      β : Type v'
      γ : Type u_1
      ⊢ Function.RightInverse FirstOrder.Language.Term.varsToConstants FirstOrder.La …
    -/
    intro t
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      α : Type u'
      β : Type v'
      γ : Type u_1
      t : L.Term (Sum γ α)
      ⊢ Eq t.varsToConstants.constantsToVars t
    -/
    induction' t with x n f _ ih
      /-
        case var
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        α : Type u'
        β : Type v'
        γ : Type u_1
        x : Sum γ α
        ⊢ Eq (FirstOrder.Language.Term.var x).varsToConstants.constantsToVars (FirstOr …
      -/
                  /-
                    🎉 no goals
                  -/
    · cases x <;> rfl
                  /-
                    🎉 no goals
                  -/
      /-
        case func
        L : FirstOrder.Language
        L' : FirstOrder.Language
        M : Type w
        α : Type u'
        β : Type v'
        γ : Type u_1
        n : Nat
        f : L.Functions n
        _ts✝ : Fin n → L.Term (Sum γ α)
        ih : ∀ (a : Fin n), Eq (_ts✝ a).varsToConstants.constantsToVars (_ts✝ a)
        ⊢ Eq (FirstOrder.Language.Term.func f _ts✝).varsToConstants.constantsToVars (F …
      -/
                    /-
                      🎉 no goals
                    -/
    · cases n <;> · simp [varsToConstants, constantsToVars, ih]⟩
                    /-
                      🎉 no goals
                    -/


/-- A bijection between terms with constants and terms with extra variables. -/
def constantsVarsEquivLeft : L[[γ]].Term (α ⊕ β) ≃ L.Term ((γ ⊕ α) ⊕ β) :=
  constantsVarsEquiv.trans (relabelEquiv (Equiv.sumAssoc _ _ _)).symm


@[simp]
theorem constantsVarsEquivLeft_apply (t : L[[γ]].Term (α ⊕ β)) :
    constantsVarsEquivLeft t = (constantsToVars t).relabel (Equiv.sumAssoc _ _ _).symm :=
  rfl


@[simp]
theorem constantsVarsEquivLeft_symm_apply (t : L.Term ((γ ⊕ α) ⊕ β)) :
    constantsVarsEquivLeft.symm t = varsToConstants (t.relabel (Equiv.sumAssoc _ _ _)) :=
  rfl


instance inhabitedOfVar [Inhabited α] : Inhabited (L.Term α) :=
  ⟨var default⟩


instance inhabitedOfConstant [Inhabited L.Constants] : Inhabited (L.Term α) :=
  ⟨(default : L.Constants).term⟩


/-- Raises all of the `Fin`-indexed variables of a term greater than or equal to `m` by `n'`. -/
def liftAt {n : ℕ} (n' m : ℕ) : L.Term (α ⊕ (Fin n)) → L.Term (α ⊕ (Fin (n + n'))) :=
  relabel (Sum.map id fun i => if ↑i < m then Fin.castAdd n' i else Fin.addNat i n')

-- Porting note: universes in different order

/-- Substitutes the variables in a given term with terms. -/
@[simp]
def subst : L.Term α → (α → L.Term β) → L.Term β
  | var a, tf => tf a
  | func f ts, tf => func f fun i => (ts i).subst tf


scoped[FirstOrder] prefix:arg "&" => FirstOrder.Language.Term.var ∘ Sum.inr


/-- Maps a term's symbols along a language map. -/
@[simp]
def onTerm (φ : L →ᴸ L') : L.Term α → L'.Term α
  | var i => var i
  | func f ts => func (φ.onFunction f) fun i => onTerm φ (ts i)


@[simp]
theorem id_onTerm : ((LHom.id L).onTerm : L.Term α → L.Term α) = id := by
  /-
    L : FirstOrder.Language
    α : Type u'
    ⊢ Eq (FirstOrder.Language.LHom.id L).onTerm id
  -/
  ext t
  induction t with
  | var => rfl
  | func _ _ ih => simp_rw [onTerm, ih]; rfl


@[simp]
theorem comp_onTerm {L'' : Language} (φ : L' →ᴸ L'') (ψ : L →ᴸ L') :
    ((φ.comp ψ).onTerm : L.Term α → L''.Term α) = φ.onTerm ∘ ψ.onTerm := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    α : Type u'
    L'' : FirstOrder.Language
    φ : L'.LHom L''
    ψ : L.LHom L'
    ⊢ Eq (φ.comp ψ).onTerm (Function.comp φ.onTerm ψ.onTerm)
  -/
  ext t
  induction t with
  | var => rfl
  | func _ _ ih => simp_rw [onTerm, ih]; rfl


/-- Maps a term's symbols along a language equivalence. -/
@[simps]
def Lequiv.onTerm (φ : L ≃ᴸ L') : L.Term α ≃ L'.Term α where
  toFun := φ.toLHom.onTerm
  invFun := φ.invLHom.onTerm
  left_inv := by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      α : Type u'
      β : Type v'
      γ : Type u_1
      φ : L.LEquiv L'
      ⊢ Function.LeftInverse φ.invLHom.onTerm φ.toLHom.onTerm
    -/
    rw [Function.leftInverse_iff_comp, ← LHom.comp_onTerm, φ.left_inv, LHom.id_onTerm]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      α : Type u'
      β : Type v'
      γ : Type u_1
      φ : L.LEquiv L'
      ⊢ Function.RightInverse φ.invLHom.onTerm φ.toLHom.onTerm
    -/
    rw [Function.rightInverse_iff_comp, ← LHom.comp_onTerm, φ.right_inv, LHom.id_onTerm]
    /-
      🎉 no goals
    -/


/-- `BoundedFormula α n` is the type of formulas with free variables indexed by `α` and up to `n`
  additional free variables. -/
inductive BoundedFormula : ℕ → Type max u v u'
  | falsum {n} : BoundedFormula n
  | equal {n} (t₁ t₂ : L.Term (α ⊕ (Fin n))) : BoundedFormula n
  | rel {n l : ℕ} (R : L.Relations l) (ts : Fin l → L.Term (α ⊕ (Fin n))) : BoundedFormula n
  | imp {n} (f₁ f₂ : BoundedFormula n) : BoundedFormula n
  | all {n} (f : BoundedFormula (n + 1)) : BoundedFormula n


/-- `Formula α` is the type of formulas with all free variables indexed by `α`. -/
abbrev Formula :=
  L.BoundedFormula α 0


/-- A sentence is a formula with no free variables. -/
abbrev Sentence :=
  L.Formula Empty


/-- A theory is a set of sentences. -/
abbrev Theory :=
  Set L.Sentence


/-- Applies a relation to terms as a bounded formula. -/
def Relations.boundedFormula {l : ℕ} (R : L.Relations n) (ts : Fin n → L.Term (α ⊕ (Fin l))) :
    L.BoundedFormula α l :=
  BoundedFormula.rel R ts


/-- Applies a unary relation to a term as a bounded formula. -/
def Relations.boundedFormula₁ (r : L.Relations 1) (t : L.Term (α ⊕ (Fin n))) :
    L.BoundedFormula α n :=
  r.boundedFormula ![t]


/-- Applies a binary relation to two terms as a bounded formula. -/
def Relations.boundedFormula₂ (r : L.Relations 2) (t₁ t₂ : L.Term (α ⊕ (Fin n))) :
    L.BoundedFormula α n :=
  r.boundedFormula ![t₁, t₂]


/-- The equality of two terms as a bounded formula. -/
def Term.bdEqual (t₁ t₂ : L.Term (α ⊕ (Fin n))) : L.BoundedFormula α n :=
  BoundedFormula.equal t₁ t₂


/-- Applies a relation to terms as a bounded formula. -/
def Relations.formula (R : L.Relations n) (ts : Fin n → L.Term α) : L.Formula α :=
  R.boundedFormula fun i => (ts i).relabel Sum.inl


/-- Applies a unary relation to a term as a formula. -/
def Relations.formula₁ (r : L.Relations 1) (t : L.Term α) : L.Formula α :=
  r.formula ![t]


/-- Applies a binary relation to two terms as a formula. -/
def Relations.formula₂ (r : L.Relations 2) (t₁ t₂ : L.Term α) : L.Formula α :=
  r.formula ![t₁, t₂]


/-- The equality of two terms as a first-order formula. -/
def Term.equal (t₁ t₂ : L.Term α) : L.Formula α :=
  (t₁.relabel Sum.inl).bdEqual (t₂.relabel Sum.inl)


instance : Inhabited (L.BoundedFormula α n) :=
  ⟨falsum⟩


instance : Bot (L.BoundedFormula α n) :=
  ⟨falsum⟩


/-- The negation of a bounded formula is also a bounded formula. -/
@[match_pattern]
protected def not (φ : L.BoundedFormula α n) : L.BoundedFormula α n :=
  φ.imp ⊥


/-- Puts an `∃` quantifier on a bounded formula. -/
@[match_pattern]
protected def ex (φ : L.BoundedFormula α (n + 1)) : L.BoundedFormula α n :=
  φ.not.all.not


instance : Top (L.BoundedFormula α n) :=
  ⟨BoundedFormula.not ⊥⟩


instance : Min (L.BoundedFormula α n) :=
  ⟨fun f g => (f.imp g.not).not⟩


instance : Max (L.BoundedFormula α n) :=
  ⟨fun f g => f.not.imp g⟩


/-- The biimplication between two bounded formulas. -/
protected def iff (φ ψ : L.BoundedFormula α n) :=
  φ.imp ψ ⊓ ψ.imp φ


/-- The `Finset` of variables used in a given formula. -/
@[simp]
def freeVarFinset [DecidableEq α] : ∀ {n}, L.BoundedFormula α n → Finset α
  | _n, falsum => ∅
  | _n, equal t₁ t₂ => t₁.varFinsetLeft ∪ t₂.varFinsetLeft
  | _n, rel _R ts => univ.biUnion fun i => (ts i).varFinsetLeft
  | _n, imp f₁ f₂ => f₁.freeVarFinset ∪ f₂.freeVarFinset
  | _n, all f => f.freeVarFinset

-- Porting note: universes in different order

/-- Casts `L.BoundedFormula α m` as `L.BoundedFormula α n`, where `m ≤ n`. -/
@[simp]
def castLE : ∀ {m n : ℕ} (_h : m ≤ n), L.BoundedFormula α m → L.BoundedFormula α n
  | _m, _n, _h, falsum => falsum
  | _m, _n, h, equal t₁ t₂ =>
    equal (t₁.relabel (Sum.map id (Fin.castLE h))) (t₂.relabel (Sum.map id (Fin.castLE h)))
  | _m, _n, h, rel R ts => rel R (Term.relabel (Sum.map id (Fin.castLE h)) ∘ ts)
  | _m, _n, h, imp f₁ f₂ => (f₁.castLE h).imp (f₂.castLE h)
  | _m, _n, h, all f => (f.castLE (add_le_add_right h 1)).all


@[simp]
theorem castLE_rfl {n} (h : n ≤ n) (φ : L.BoundedFormula α n) : φ.castLE h = φ := by
  induction φ with
  | falsum => rfl
  | equal => simp [Fin.castLE_of_eq]
  | rel => simp [Fin.castLE_of_eq]
  | imp _ _ ih1 ih2 => simp [Fin.castLE_of_eq, ih1, ih2]
  | all _ ih3 => simp [Fin.castLE_of_eq, ih3]


@[simp]
theorem castLE_castLE {k m n} (km : k ≤ m) (mn : m ≤ n) (φ : L.BoundedFormula α k) :
    (φ.castLE km).castLE mn = φ.castLE (km.trans mn) := by
  /-
    L : FirstOrder.Language
    α : Type u'
    k m n : Nat
    km : LE.le k m
    mn : LE.le m n
    φ : L.BoundedFormula α k
    ⊢ Eq (FirstOrder.Language.BoundedFormula.castLE mn (FirstOrder.Language.Bounde …
  -/
  revert m n
  induction φ with
  | falsum => intros; rfl
  | equal => simp
  | rel =>
    intros
    simp only [castLE, eq_self_iff_true, heq_iff_eq]
    rw [← Function.comp_assoc, Term.relabel_comp_relabel]
    simp
  | imp _ _ ih1 ih2 => simp [ih1, ih2]
  | all _ ih3 => intros; simp only [castLE, ih3]


@[simp]
theorem castLE_comp_castLE {k m n} (km : k ≤ m) (mn : m ≤ n) :
    (BoundedFormula.castLE mn ∘ BoundedFormula.castLE km :
        L.BoundedFormula α k → L.BoundedFormula α n) =
      BoundedFormula.castLE (km.trans mn) :=
  funext (castLE_castLE km mn)

-- Porting note: universes in different order

/-- Restricts a bounded formula to only use a particular set of free variables. -/
def restrictFreeVar [DecidableEq α] :
    ∀ {n : ℕ} (φ : L.BoundedFormula α n) (_f : φ.freeVarFinset → β), L.BoundedFormula β n
  | _n, falsum, _f => falsum
  | _n, equal t₁ t₂, f =>
    equal (t₁.restrictVarLeft (f ∘ Set.inclusion subset_union_left))
      (t₂.restrictVarLeft (f ∘ Set.inclusion subset_union_right))
  | _n, rel R ts, f =>
    rel R fun i => (ts i).restrictVarLeft (f ∘ Set.inclusion
      (subset_biUnion_of_mem (fun i => Term.varFinsetLeft (ts i)) (mem_univ i)))
  | _n, imp φ₁ φ₂, f =>
    (φ₁.restrictFreeVar (f ∘ Set.inclusion subset_union_left)).imp
      (φ₂.restrictFreeVar (f ∘ Set.inclusion subset_union_right))
  | _n, all φ, f => (φ.restrictFreeVar f).all

-- Porting note: universes in different order

/-- Places universal quantifiers on all extra variables of a bounded formula. -/
def alls : ∀ {n}, L.BoundedFormula α n → L.Formula α
  | 0, φ => φ
  | _n + 1, φ => φ.all.alls

-- Porting note: universes in different order

/-- Places existential quantifiers on all extra variables of a bounded formula. -/
def exs : ∀ {n}, L.BoundedFormula α n → L.Formula α
  | 0, φ => φ
  | _n + 1, φ => φ.ex.exs

-- Porting note: universes in different order

/-- Maps bounded formulas along a map of terms and a map of relations. -/
def mapTermRel {g : ℕ → ℕ} (ft : ∀ n, L.Term (α ⊕ (Fin n)) → L'.Term (β ⊕ (Fin (g n))))
    (fr : ∀ n, L.Relations n → L'.Relations n)
    (h : ∀ n, L'.BoundedFormula β (g (n + 1)) → L'.BoundedFormula β (g n + 1)) :
    ∀ {n}, L.BoundedFormula α n → L'.BoundedFormula β (g n)
  | _n, falsum => falsum
  | _n, equal t₁ t₂ => equal (ft _ t₁) (ft _ t₂)
  | _n, rel R ts => rel (fr _ R) fun i => ft _ (ts i)
  | _n, imp φ₁ φ₂ => (φ₁.mapTermRel ft fr h).imp (φ₂.mapTermRel ft fr h)
  | n, all φ => (h n (φ.mapTermRel ft fr h)).all


/-- Raises all of the `Fin`-indexed variables of a formula greater than or equal to `m` by `n'`. -/
def liftAt : ∀ {n : ℕ} (n' _m : ℕ), L.BoundedFormula α n → L.BoundedFormula α (n + n') :=
  fun {_} n' m φ =>
  φ.mapTermRel (fun _ t => t.liftAt n' m) (fun _ => id) fun _ =>
               /-
                 L : FirstOrder.Language
                 L' : FirstOrder.Language
                 M : Type w
                 α : Type u'
                 β : Type v'
                 γ : Type u_1
                 n x✝¹ n' m : Nat
                 φ : L.BoundedFormula α x✝¹
                 x✝ : Nat
                 ⊢ LE.le (HAdd.hAdd (HAdd.hAdd x✝ 1) n') (HAdd.hAdd (HAdd.hAdd x✝ n') 1)
               -/
    castLE (by rw [add_assoc, add_comm 1, add_assoc])
               /-
                 🎉 no goals
               -/


@[simp]
theorem mapTermRel_mapTermRel {L'' : Language}
    (ft : ∀ n, L.Term (α ⊕ (Fin n)) → L'.Term (β ⊕ (Fin n)))
    (fr : ∀ n, L.Relations n → L'.Relations n)
    (ft' : ∀ n, L'.Term (β ⊕ Fin n) → L''.Term (γ ⊕ (Fin n)))
    (fr' : ∀ n, L'.Relations n → L''.Relations n) {n} (φ : L.BoundedFormula α n) :
    ((φ.mapTermRel ft fr fun _ => id).mapTermRel ft' fr' fun _ => id) =
      φ.mapTermRel (fun _ => ft' _ ∘ ft _) (fun _ => fr' _ ∘ fr _) fun _ => id := by
  induction φ with
  | falsum => rfl
  | equal => simp [mapTermRel]
  | rel => simp [mapTermRel]
  | imp _ _ ih1 ih2 => simp [mapTermRel, ih1, ih2]
  | all _ ih3 => simp [mapTermRel, ih3]


@[simp]
theorem mapTermRel_id_id_id {n} (φ : L.BoundedFormula α n) :
    (φ.mapTermRel (fun _ => id) (fun _ => id) fun _ => id) = φ := by
  induction φ with
  | falsum => rfl
  | equal => simp [mapTermRel]
  | rel => simp [mapTermRel]
  | imp _ _ ih1 ih2 => simp [mapTermRel, ih1, ih2]
  | all _ ih3 => simp [mapTermRel, ih3]


/-- An equivalence of bounded formulas given by an equivalence of terms and an equivalence of
relations. -/
@[simps]
def mapTermRelEquiv (ft : ∀ n, L.Term (α ⊕ (Fin n)) ≃ L'.Term (β ⊕ (Fin n)))
    (fr : ∀ n, L.Relations n ≃ L'.Relations n) {n} : L.BoundedFormula α n ≃ L'.BoundedFormula β n :=
  ⟨mapTermRel (fun n => ft n) (fun n => fr n) fun _ => id,
                                                                                      /-
                                                                                        L : FirstOrder.Language
                                                                                        L' : FirstOrder.Language
                                                                                        M : Type w
                                                                                        α : Type u'
                                                                                        β : Type v'
                                                                                        γ : Type u_1
                                                                                        n✝ : Nat
                                                                                        ft : (n : Nat) → _root_.Equiv (L.Term (Sum α (Fin n))) (L'.Term (Sum β (Fin n)))
                                                                                        fr : (n : Nat) → _root_.Equiv (L.Relations n) (L'.Relations n)
                                                                                        n : Nat
                                                                                        φ : L.BoundedFormula α n
                                                                                        ⊢ Eq (FirstOrder.Language.BoundedFormula.mapTermRel (fun n => ⇑(ft n).symm) (f …
                                                                                      -/
    mapTermRel (fun n => (ft n).symm) (fun n => (fr n).symm) fun _ => id, fun φ => by simp, fun φ =>
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
       /-
         L : FirstOrder.Language
         L' : FirstOrder.Language
         M : Type w
         α : Type u'
         β : Type v'
         γ : Type u_1
         n✝ : Nat
         ft : (n : Nat) → _root_.Equiv (L.Term (Sum α (Fin n))) (L'.Term (Sum β (Fin n)))
         fr : (n : Nat) → _root_.Equiv (L.Relations n) (L'.Relations n)
         n : Nat
         φ : L'.BoundedFormula β n
         ⊢ Eq (FirstOrder.Language.BoundedFormula.mapTermRel (fun n => ⇑(ft n)) (fun n  …
       -/
    by simp⟩
       /-
         🎉 no goals
       -/


/-- A function to help relabel the variables in bounded formulas. -/
def relabelAux (g : α → β ⊕ (Fin n)) (k : ℕ) : α ⊕ (Fin k) → β ⊕ (Fin (n + k)) :=
  Sum.map id finSumFinEquiv ∘ Equiv.sumAssoc _ _ _ ∘ Sum.map g id


@[simp]
theorem sum_elim_comp_relabelAux {m : ℕ} {g : α → β ⊕ (Fin n)} {v : β → M}
    {xs : Fin (n + m) → M} : Sum.elim v xs ∘ relabelAux g m =
    Sum.elim (Sum.elim v (xs ∘ castAdd m) ∘ g) (xs ∘ natAdd n) := by
  /-
    M : Type w
    α : Type u'
    β : Type v'
    n m : Nat
    g : α → Sum β (Fin n)
    v : β → M
    xs : Fin (HAdd.hAdd n m) → M
    ⊢ Eq (Function.comp (Sum.elim v xs) (FirstOrder.Language.BoundedFormula.relabe …
  -/
  ext x
  /-
    case h
    M : Type w
    α : Type u'
    β : Type v'
    n m : Nat
    g : α → Sum β (Fin n)
    v : β → M
    xs : Fin (HAdd.hAdd n m) → M
    x : Sum α (Fin m)
    ⊢ Eq (Function.comp (Sum.elim v xs) (FirstOrder.Language.BoundedFormula.relabe …
  -/
  cases' x with x x
    /-
      case h.inl
      M : Type w
      α : Type u'
      β : Type v'
      n m : Nat
      g : α → Sum β (Fin n)
      v : β → M
      xs : Fin (HAdd.hAdd n m) → M
      x : α
      ⊢ Eq (Function.comp (Sum.elim v xs) (FirstOrder.Language.BoundedFormula.relabe …
    -/
  · simp only [BoundedFormula.relabelAux, Function.comp_apply, Sum.map_inl, Sum.elim_inl]
    /-
      case h.inl
      M : Type w
      α : Type u'
      β : Type v'
      n m : Nat
      g : α → Sum β (Fin n)
      v : β → M
      xs : Fin (HAdd.hAdd n m) → M
      x : α
      ⊢ Eq (Sum.elim v xs (Sum.map id (⇑finSumFinEquiv) ((Equiv.sumAssoc β (Fin n) ( …
    -/
                            /-
                              🎉 no goals
                            -/
    cases' g x with l r <;> simp
                            /-
                              🎉 no goals
                            -/
    /-
      case h.inr
      M : Type w
      α : Type u'
      β : Type v'
      n m : Nat
      g : α → Sum β (Fin n)
      v : β → M
      xs : Fin (HAdd.hAdd n m) → M
      x : Fin m
      ⊢ Eq (Function.comp (Sum.elim v xs) (FirstOrder.Language.BoundedFormula.relabe …
    -/
  · simp [BoundedFormula.relabelAux]
    /-
      🎉 no goals
    -/


@[simp]
theorem relabelAux_sum_inl (k : ℕ) :
    relabelAux (Sum.inl : α → α ⊕ (Fin n)) k = Sum.map id (natAdd n) := by
  /-
    α : Type u'
    n k : Nat
    ⊢ Eq (FirstOrder.Language.BoundedFormula.relabelAux Sum.inl k) (Sum.map id (Fi …
  -/
  ext x
  /-
    case h
    α : Type u'
    n k : Nat
    x : Sum α (Fin k)
    ⊢ Eq (FirstOrder.Language.BoundedFormula.relabelAux Sum.inl k x) (Sum.map id ( …
  -/
                /-
                  🎉 no goals
                -/
  cases x <;> · simp [relabelAux]
                /-
                  🎉 no goals
                -/


/-- Relabels a bounded formula's variables along a particular function. -/
def relabel (g : α → β ⊕ (Fin n)) {k} (φ : L.BoundedFormula α k) : L.BoundedFormula β (n + k) :=
  φ.mapTermRel (fun _ t => t.relabel (relabelAux g _)) (fun _ => id) fun _ =>
    castLE (ge_of_eq (add_assoc _ _ _))


/-- Relabels a bounded formula's free variables along a bijection. -/
def relabelEquiv (g : α ≃ β) {k} : L.BoundedFormula α k ≃ L.BoundedFormula β k :=
  mapTermRelEquiv (fun _n => Term.relabelEquiv (g.sumCongr (_root_.Equiv.refl _)))
    fun _n => _root_.Equiv.refl _


@[simp]
theorem relabel_falsum (g : α → β ⊕ (Fin n)) {k} :
    (falsum : L.BoundedFormula α k).relabel g = falsum :=
  rfl


@[simp]
theorem relabel_bot (g : α → β ⊕ (Fin n)) {k} : (⊥ : L.BoundedFormula α k).relabel g = ⊥ :=
  rfl


@[simp]
theorem relabel_imp (g : α → β ⊕ (Fin n)) {k} (φ ψ : L.BoundedFormula α k) :
    (φ.imp ψ).relabel g = (φ.relabel g).imp (ψ.relabel g) :=
  rfl


@[simp]
theorem relabel_not (g : α → β ⊕ (Fin n)) {k} (φ : L.BoundedFormula α k) :
                                              /-
                                                L : FirstOrder.Language
                                                α : Type u'
                                                β : Type v'
                                                n : Nat
                                                g : α → Sum β (Fin n)
                                                k : Nat
                                                φ : L.BoundedFormula α k
                                                ⊢ Eq (FirstOrder.Language.BoundedFormula.relabel g φ.not) (FirstOrder.Language …
                                              -/
    φ.not.relabel g = (φ.relabel g).not := by simp [BoundedFormula.not]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem relabel_all (g : α → β ⊕ (Fin n)) {k} (φ : L.BoundedFormula α (k + 1)) :
    φ.all.relabel g = (φ.relabel g).all := by
  /-
    L : FirstOrder.Language
    α : Type u'
    β : Type v'
    n : Nat
    g : α → Sum β (Fin n)
    k : Nat
    φ : L.BoundedFormula α (HAdd.hAdd k 1)
    ⊢ Eq (FirstOrder.Language.BoundedFormula.relabel g φ.all) (FirstOrder.Language …
  -/
  rw [relabel, mapTermRel, relabel]
  /-
    L : FirstOrder.Language
    α : Type u'
    β : Type v'
    n : Nat
    g : α → Sum β (Fin n)
    k : Nat
    φ : L.BoundedFormula α (HAdd.hAdd k 1)
    ⊢ Eq (FirstOrder.Language.BoundedFormula.castLE ⋯ (FirstOrder.Language.Bounded …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem relabel_ex (g : α → β ⊕ (Fin n)) {k} (φ : L.BoundedFormula α (k + 1)) :
                                            /-
                                              L : FirstOrder.Language
                                              α : Type u'
                                              β : Type v'
                                              n : Nat
                                              g : α → Sum β (Fin n)
                                              k : Nat
                                              φ : L.BoundedFormula α (HAdd.hAdd k 1)
                                              ⊢ Eq (FirstOrder.Language.BoundedFormula.relabel g φ.ex) (FirstOrder.Language. …
                                            -/
    φ.ex.relabel g = (φ.relabel g).ex := by simp [BoundedFormula.ex]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem relabel_sum_inl (φ : L.BoundedFormula α n) :
    (φ.relabel Sum.inl : L.BoundedFormula α (0 + n)) = φ.castLE (ge_of_eq (zero_add n)) := by
  /-
    L : FirstOrder.Language
    α : Type u'
    n : Nat
    φ : L.BoundedFormula α n
    ⊢ Eq (FirstOrder.Language.BoundedFormula.relabel Sum.inl φ) (FirstOrder.Langua …
  -/
  simp only [relabel, relabelAux_sum_inl]
  induction φ with
  | falsum => rfl
  | equal => simp [Fin.natAdd_zero, castLE_of_eq, mapTermRel]
  | rel => simp [Fin.natAdd_zero, castLE_of_eq, mapTermRel]; rfl
  | imp _ _ ih1 ih2 => simp_all [mapTermRel]
  | all _ ih3 => simp_all [mapTermRel]


/-- Substitutes the variables in a given formula with terms. -/
def subst {n : ℕ} (φ : L.BoundedFormula α n) (f : α → L.Term β) : L.BoundedFormula β n :=
  φ.mapTermRel (fun _ t => t.subst (Sum.elim (Term.relabel Sum.inl ∘ f) (var ∘ Sum.inr)))
    (fun _ => id) fun _ => id


/-- A bijection sending formulas with constants to formulas with extra variables. -/
def constantsVarsEquiv : L[[γ]].BoundedFormula α n ≃ L.BoundedFormula (γ ⊕ α) n :=
  mapTermRelEquiv (fun _ => Term.constantsVarsEquivLeft) fun _ => Equiv.sumEmpty _ _

-- Porting note: universes in different order

/-- Turns the extra variables of a bounded formula into free variables. -/
@[simp]
def toFormula : ∀ {n : ℕ}, L.BoundedFormula α n → L.Formula (α ⊕ (Fin n))
  | _n, falsum => falsum
  | _n, equal t₁ t₂ => t₁.equal t₂
  | _n, rel R ts => R.formula ts
  | _n, imp φ₁ φ₂ => φ₁.toFormula.imp φ₂.toFormula
  | _n, all φ =>
    (φ.toFormula.relabel
        (Sum.elim (Sum.inl ∘ Sum.inl) (Sum.map Sum.inr id ∘ finSumFinEquiv.symm))).all


/-- take the disjunction of a finite set of formulas -/
noncomputable def iSup [Finite β] (f : β → L.BoundedFormula α n) : L.BoundedFormula α n :=
  let _ := Fintype.ofFinite β
  ((Finset.univ : Finset β).toList.map f).foldr (· ⊔ ·) ⊥


/-- take the conjunction of a finite set of formulas -/
noncomputable def iInf [Finite β] (f : β → L.BoundedFormula α n) : L.BoundedFormula α n :=
  let _ := Fintype.ofFinite β
  ((Finset.univ : Finset β).toList.map f).foldr (· ⊓ ·) ⊤


/-- Maps a bounded formula's symbols along a language map. -/
@[simp]
def onBoundedFormula (g : L →ᴸ L') : ∀ {k : ℕ}, L.BoundedFormula α k → L'.BoundedFormula α k
  | _k, falsum => falsum
  | _k, equal t₁ t₂ => (g.onTerm t₁).bdEqual (g.onTerm t₂)
  | _k, rel R ts => (g.onRelation R).boundedFormula (g.onTerm ∘ ts)
  | _k, imp f₁ f₂ => (onBoundedFormula g f₁).imp (onBoundedFormula g f₂)
  | _k, all f => (onBoundedFormula g f).all


@[simp]
theorem id_onBoundedFormula :
    ((LHom.id L).onBoundedFormula : L.BoundedFormula α n → L.BoundedFormula α n) = id := by
  /-
    L : FirstOrder.Language
    α : Type u'
    n : Nat
    ⊢ Eq (FirstOrder.Language.LHom.id L).onBoundedFormula id
  -/
  ext f
  induction f with
  | falsum => rfl
  | equal => rw [onBoundedFormula, LHom.id_onTerm, id, id, id, Term.bdEqual]
  | rel => rw [onBoundedFormula, LHom.id_onTerm]; rfl
  | imp _ _ ih1 ih2 => rw [onBoundedFormula, ih1, ih2, id, id, id]
  | all _ ih3 => rw [onBoundedFormula, ih3, id, id]


@[simp]
theorem comp_onBoundedFormula {L'' : Language} (φ : L' →ᴸ L'') (ψ : L →ᴸ L') :
    ((φ.comp ψ).onBoundedFormula : L.BoundedFormula α n → L''.BoundedFormula α n) =
      φ.onBoundedFormula ∘ ψ.onBoundedFormula := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    α : Type u'
    n : Nat
    L'' : FirstOrder.Language
    φ : L'.LHom L''
    ψ : L.LHom L'
    ⊢ Eq (φ.comp ψ).onBoundedFormula (Function.comp φ.onBoundedFormula ψ.onBounded …
  -/
  ext f
  induction f with
  | falsum => rfl
  | equal => simp [Term.bdEqual]
  | rel => simp only [onBoundedFormula, comp_onRelation, comp_onTerm, Function.comp_apply]; rfl
  | imp _ _ ih1 ih2 =>
    simp only [onBoundedFormula, Function.comp_apply, ih1, ih2, eq_self_iff_true, and_self_iff]
  | all _ ih3 => simp only [ih3, onBoundedFormula, Function.comp_apply]


/-- Maps a formula's symbols along a language map. -/
def onFormula (g : L →ᴸ L') : L.Formula α → L'.Formula α :=
  g.onBoundedFormula


/-- Maps a sentence's symbols along a language map. -/
def onSentence (g : L →ᴸ L') : L.Sentence → L'.Sentence :=
  g.onFormula


/-- Maps a theory's symbols along a language map. -/
def onTheory (g : L →ᴸ L') (T : L.Theory) : L'.Theory :=
  g.onSentence '' T


@[simp]
theorem mem_onTheory {g : L →ᴸ L'} {T : L.Theory} {φ : L'.Sentence} :
    φ ∈ g.onTheory T ↔ ∃ φ₀, φ₀ ∈ T ∧ g.onSentence φ₀ = φ :=
  Set.mem_image _ _ _


/-- Maps a bounded formula's symbols along a language equivalence. -/
@[simps]
def onBoundedFormula (φ : L ≃ᴸ L') : L.BoundedFormula α n ≃ L'.BoundedFormula α n where
  toFun := φ.toLHom.onBoundedFormula
  invFun := φ.invLHom.onBoundedFormula
  left_inv := by
    rw [Function.leftInverse_iff_comp, ← LHom.comp_onBoundedFormula, φ.left_inv,
      LHom.id_onBoundedFormula]
  right_inv := by
    rw [Function.rightInverse_iff_comp, ← LHom.comp_onBoundedFormula, φ.right_inv,
      LHom.id_onBoundedFormula]


theorem onBoundedFormula_symm (φ : L ≃ᴸ L') :
    (φ.onBoundedFormula.symm : L'.BoundedFormula α n ≃ L.BoundedFormula α n) =
      φ.symm.onBoundedFormula :=
  rfl


/-- Maps a formula's symbols along a language equivalence. -/
def onFormula (φ : L ≃ᴸ L') : L.Formula α ≃ L'.Formula α :=
  φ.onBoundedFormula


@[simp]
theorem onFormula_apply (φ : L ≃ᴸ L') :
    (φ.onFormula : L.Formula α → L'.Formula α) = φ.toLHom.onFormula :=
  rfl


@[simp]
theorem onFormula_symm (φ : L ≃ᴸ L') :
    (φ.onFormula.symm : L'.Formula α ≃ L.Formula α) = φ.symm.onFormula :=
  rfl


/-- Maps a sentence's symbols along a language equivalence. -/
@[simps!] -- Porting note: add `!` to `simps`
def onSentence (φ : L ≃ᴸ L') : L.Sentence ≃ L'.Sentence :=
  φ.onFormula


@[inherit_doc] scoped[FirstOrder] infixl:88 " =' " => FirstOrder.Language.Term.bdEqual
-- input \~- or \simeq


scoped[FirstOrder] infixr:62 " ⟹ " => FirstOrder.Language.BoundedFormula.imp
-- input \==>


scoped[FirstOrder] prefix:110 "∀'" => FirstOrder.Language.BoundedFormula.all


@[inherit_doc] scoped[FirstOrder] prefix:arg "∼" => FirstOrder.Language.BoundedFormula.not
-- input \~, the ASCII character ~ has too low precedence


@[inherit_doc] scoped[FirstOrder] infixl:61 " ⇔ " => FirstOrder.Language.BoundedFormula.iff
-- input \<=>


@[inherit_doc] scoped[FirstOrder] prefix:110 "∃'" => FirstOrder.Language.BoundedFormula.ex
-- input \ex


/-- Relabels a formula's variables along a particular function. -/
def relabel (g : α → β) : L.Formula α → L.Formula β :=
  @BoundedFormula.relabel _ _ _ 0 (Sum.inl ∘ g) 0


/-- The graph of a function as a first-order formula. -/
def graph (f : L.Functions n) : L.Formula (Fin (n + 1)) :=
  Term.equal (var 0) (func f fun i => var i.succ)


/-- The negation of a formula. -/
protected nonrec abbrev not (φ : L.Formula α) : L.Formula α :=
  φ.not


/-- The implication between formulas, as a formula. -/
protected abbrev imp : L.Formula α → L.Formula α → L.Formula α :=
  BoundedFormula.imp



variable (β) in
/-- `iAlls f φ` transforms a `L.Formula (α ⊕ β)` into a `L.Formula β` by universally
quantifying over all variables `Sum.inr _`. -/
noncomputable def iAlls [Finite β] (φ : L.Formula (α ⊕ β)) : L.Formula α :=
  let e := Classical.choice (Classical.choose_spec (Finite.exists_equiv_fin β))
  (BoundedFormula.relabel (fun a => Sum.map id e a) φ).alls


variable (β) in
/-- `iExs f φ` transforms a `L.Formula (α ⊕ β)` into a `L.Formula β` by existentially
quantifying over all variables `Sum.inr _`. -/
noncomputable def iExs [Finite β] (φ : L.Formula (α ⊕ β)) : L.Formula α :=
  let e := Classical.choice (Classical.choose_spec (Finite.exists_equiv_fin β))
  (BoundedFormula.relabel (fun a => Sum.map id e a) φ).exs


/-- The biimplication between formulas, as a formula. -/
protected nonrec abbrev iff (φ ψ : L.Formula α) : L.Formula α :=
  φ.iff ψ


/-- A bijection sending formulas to sentences with constants. -/
def equivSentence : L.Formula α ≃ L[[α]].Sentence :=
  (BoundedFormula.constantsVarsEquiv.trans (BoundedFormula.relabelEquiv (Equiv.sumEmpty _ _))).symm


theorem equivSentence_not (φ : L.Formula α) : equivSentence φ.not = (equivSentence φ).not :=
  rfl


theorem equivSentence_inf (φ ψ : L.Formula α) :
    equivSentence (φ ⊓ ψ) = equivSentence φ ⊓ equivSentence ψ :=
  rfl


/-- The sentence indicating that a basic relation symbol is reflexive. -/
protected def reflexive : L.Sentence :=
  ∀'r.boundedFormula₂ (&0) &0


/-- The sentence indicating that a basic relation symbol is irreflexive. -/
protected def irreflexive : L.Sentence :=
  ∀'∼(r.boundedFormula₂ (&0) &0)


/-- The sentence indicating that a basic relation symbol is symmetric. -/
protected def symmetric : L.Sentence :=
  ∀'∀'(r.boundedFormula₂ (&0) &1 ⟹ r.boundedFormula₂ (&1) &0)


/-- The sentence indicating that a basic relation symbol is antisymmetric. -/
protected def antisymmetric : L.Sentence :=
  ∀'∀'(r.boundedFormula₂ (&0) &1 ⟹ r.boundedFormula₂ (&1) &0 ⟹ Term.bdEqual (&0) &1)


/-- The sentence indicating that a basic relation symbol is transitive. -/
protected def transitive : L.Sentence :=
  ∀'∀'∀'(r.boundedFormula₂ (&0) &1 ⟹ r.boundedFormula₂ (&1) &2 ⟹ r.boundedFormula₂ (&0) &2)


/-- The sentence indicating that a basic relation symbol is total. -/
protected def total : L.Sentence :=
  ∀'∀'(r.boundedFormula₂ (&0) &1 ⊔ r.boundedFormula₂ (&1) &0)


/-- A sentence indicating that a structure has `n` distinct elements. -/
protected def Sentence.cardGe (n : ℕ) : L.Sentence :=
  ((((List.finRange n ×ˢ List.finRange n).filter fun ij : _ × _ => ij.1 ≠ ij.2).map
          fun ij : _ × _ => ∼((&ij.1).bdEqual &ij.2)).foldr
      (· ⊓ ·) ⊤).exs


/-- A theory indicating that a structure is infinite. -/
def infiniteTheory : L.Theory :=
  Set.range (Sentence.cardGe L)


/-- A theory that indicates a structure is nonempty. -/
def nonemptyTheory : L.Theory :=
  {Sentence.cardGe L 1}


/-- A theory indicating that each of a set of constants is distinct. -/
def distinctConstantsTheory (s : Set α) : L[[α]].Theory :=
  (fun ab : α × α => ((L.con ab.1).term.equal (L.con ab.2).term).not) ''
  (s ×ˢ s ∩ (Set.diagonal α)ᶜ)


theorem distinctConstantsTheory_mono {s t : Set α} (h : s ⊆ t) :
    L.distinctConstantsTheory s ⊆ L.distinctConstantsTheory t := by
  /-
    L : FirstOrder.Language
    α : Type u'
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset (L.distinctConstantsTheory s) (L.distinctConstantsTheory t)
  -/
  unfold distinctConstantsTheory; gcongr
                                  /-
                                    🎉 no goals
                                  -/


theorem monotone_distinctConstantsTheory :
    Monotone (L.distinctConstantsTheory : Set α → L[[α]].Theory) := fun _s _t st =>
  L.distinctConstantsTheory_mono st


theorem directed_distinctConstantsTheory :
    Directed (· ⊆ ·) (L.distinctConstantsTheory : Set α → L[[α]].Theory) :=
  Monotone.directed_le monotone_distinctConstantsTheory


theorem distinctConstantsTheory_eq_iUnion (s : Set α) :
    L.distinctConstantsTheory s =
      ⋃ t : Finset s,
        L.distinctConstantsTheory (t.map (Function.Embedding.subtype fun x => x ∈ s)) := by
  classical
    simp only [distinctConstantsTheory]
    rw [← image_iUnion, ← iUnion_inter]
    refine congr(_ '' ($(?_) ∩ _))
    ext ⟨i, j⟩
    simp only [prod_mk_mem_set_prod_eq, Finset.coe_map, Function.Embedding.coe_subtype, mem_iUnion,
      mem_image, Finset.mem_coe, Subtype.exists, Subtype.coe_mk, exists_and_right, exists_eq_right]
    refine ⟨fun h => ⟨{⟨i, h.1⟩, ⟨j, h.2⟩}, ⟨h.1, ?_⟩, ⟨h.2, ?_⟩⟩, ?_⟩
    · simp
    · simp
    · rintro ⟨t, ⟨is, _⟩, ⟨js, _⟩⟩
      exact ⟨is, js⟩


