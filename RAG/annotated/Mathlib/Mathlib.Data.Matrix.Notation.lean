/-- Matrices can be reflected whenever their entries can. We insert a `Matrix.of` to
prevent immediate decay to a function. -/
protected instance toExpr [ToLevel.{u}] [ToLevel.{uₘ}] [ToLevel.{uₙ}]
    [Lean.ToExpr α] [Lean.ToExpr m'] [Lean.ToExpr n'] [Lean.ToExpr (m' → n' → α)] :
    Lean.ToExpr (Matrix m' n' α) :=
  have eα : Q(Type $(toLevel.{u})) := toTypeExpr α
  have em' : Q(Type $(toLevel.{uₘ})) := toTypeExpr m'
  have en' : Q(Type $(toLevel.{uₙ})) := toTypeExpr n'
  { toTypeExpr :=
    q(Matrix $eα $em' $en')
    toExpr := fun M =>
      have eM : Q($em' → $en' → $eα) := toExpr (show m' → n' → α from M)
      q(Matrix.of $eM) }


/-- Notation for m×n matrices, aka `Matrix (Fin m) (Fin n) α`.

For instance:
* `!![a, b, c; d, e, f]` is the matrix with two rows and three columns, of type
  `Matrix (Fin 2) (Fin 3) α`
* `!![a, b, c]` is a row vector of type `Matrix (Fin 1) (Fin 3) α` (see also `Matrix.row`).
* `!![a; b; c]` is a column vector of type `Matrix (Fin 3) (Fin 1) α` (see also `Matrix.col`).

This notation implements some special cases:

* `![,,]`, with `n` `,`s, is a term of type `Matrix (Fin 0) (Fin n) α`
* `![;;]`, with `m` `;`s, is a term of type `Matrix (Fin m) (Fin 0) α`
* `![]` is the 0×0 matrix

Note that vector notation is provided elsewhere (by `Matrix.vecNotation`) as `![a, b, c]`.
Under the hood, `!![a, b, c; d, e, f]` is syntax for `Matrix.of ![![a, b, c], ![d, e, f]]`.
-/
syntax (name := matrixNotation)
  "!![" ppRealGroup(sepBy1(ppGroup(term,+,?), ";", "; ", allowTrailingSep)) "]" : term


@[inherit_doc matrixNotation]
syntax (name := matrixNotationRx0) "!![" ";"+ "]" : term

@[inherit_doc matrixNotation]
syntax (name := matrixNotation0xC) "!![" ","* "]" : term


macro_rules
  | `(!![$[$[$rows],*];*]) => do
    let m := rows.size
    let n := if h : 0 < m then rows[0].size else 0
    let rowVecs ← rows.mapM fun row : Array Term => do
      unless row.size = n do
        Macro.throwErrorAt (mkNullNode row) s!"\
          Rows must be of equal length; this row has {row.size} items, \
          the previous rows have {n}"
      `(![$row,*])
    `(@Matrix.of (Fin $(quote m)) (Fin $(quote n)) _ ![$rowVecs,*])
  | `(!![$[;%$semicolons]*]) => do
    let emptyVec ← `(![])
    let emptyVecs := semicolons.map (fun _ => emptyVec)
    `(@Matrix.of (Fin $(quote semicolons.size)) (Fin 0) _ ![$emptyVecs,*])
  | `(!![$[,%$commas]*]) => `(@Matrix.of (Fin 0) (Fin $(quote commas.size)) _ ![])


/-- Delaborator for the `!![]` notation. -/
@[app_delab DFunLike.coe]
def delabMatrixNotation : Delab := whenNotPPOption getPPExplicit <| whenPPOption getPPNotation <|
  withOverApp 6 do
    let mkApp3 (.const ``Matrix.of _) (.app (.const ``Fin _) em) (.app (.const ``Fin _) en) _ :=
      (← getExpr).appFn!.appArg! | failure
    let some m ← withNatValue em (pure ∘ some) | failure
    let some n ← withNatValue en (pure ∘ some) | failure
    withAppArg do
      if m = 0 then
        guard <| (← getExpr).isAppOfArity ``vecEmpty 1
        let commas := mkArray n (mkAtom ",")
        `(!![$[,%$commas]*])
      else
        if n = 0 then
          let `(![$[![]%$evecs],*]) ← delab | failure
          `(!![$[;%$evecs]*])
        else
          let `(![$[![$[$melems],*]],*]) ← delab | failure
          `(!![$[$[$melems],*];*])


/-- Use `![...]` notation for displaying a `Fin`-indexed matrix, for example:

```
#eval !![1, 2; 3, 4] + !![3, 4; 5, 6]  -- !![4, 6; 8, 10]
```
-/
instance repr [Repr α] : Repr (Matrix (Fin m) (Fin n) α) where
  reprPrec f _p :=
    (Std.Format.bracket "!![" · "]") <|
      (Std.Format.joinSep · (";" ++ Std.Format.line)) <|
        (List.finRange m).map fun i =>
          Std.Format.fill <|  -- wrap line in a single place rather than all at once
            (Std.Format.joinSep · ("," ++ Std.Format.line)) <|
            (List.finRange n).map fun j => _root_.repr (f i j)


@[simp]
theorem cons_val' (v : n' → α) (B : Fin m → n' → α) (i j) :
                                                             /-
                                                               α : Type u
                                                               m : Nat
                                                               n' : Type uₙ
                                                               v : n' → α
                                                               B : Fin m → n' → α
                                                               i : Fin m.succ
                                                               j : n'
                                                               ⊢ Eq (Matrix.vecCons v B i j) (Matrix.vecCons (v j) (fun i => B i j) i)
                                                             -/
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    vecCons v B i j = vecCons (v j) (fun i => B i j) i := by refine Fin.cases ?_ ?_ i <;> simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
theorem head_val' (B : Fin m.succ → n' → α) (j : n') : (vecHead fun i => B i j) = vecHead B j :=
  rfl


@[simp]
theorem tail_val' (B : Fin m.succ → n' → α) (j : n') :
    (vecTail fun i => B i j) = fun i => vecTail B i j := rfl


@[simp]
theorem dotProduct_empty (v w : Fin 0 → α) : dotProduct v w = 0 :=
  Finset.sum_empty


@[simp]
theorem cons_dotProduct (x : α) (v : Fin n → α) (w : Fin n.succ → α) :
    dotProduct (vecCons x v) w = x * vecHead w + dotProduct v (vecTail w) := by
  /-
    α : Type u
    n : Nat
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    x : α
    v : Fin n → α
    w : Fin n.succ → α
    ⊢ Eq (dotProduct (Matrix.vecCons x v) w) (HAdd.hAdd (HMul.hMul x (Matrix.vecHe …
  -/
  simp [dotProduct, Fin.sum_univ_succ, vecHead, vecTail]
  /-
    🎉 no goals
  -/


@[simp]
theorem dotProduct_cons (v : Fin n.succ → α) (x : α) (w : Fin n → α) :
    dotProduct v (vecCons x w) = vecHead v * x + dotProduct (vecTail v) w := by
  /-
    α : Type u
    n : Nat
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    v : Fin n.succ → α
    x : α
    w : Fin n → α
    ⊢ Eq (dotProduct v (Matrix.vecCons x w)) (HAdd.hAdd (HMul.hMul (Matrix.vecHead …
  -/
  simp [dotProduct, Fin.sum_univ_succ, vecHead, vecTail]
  /-
    🎉 no goals
  -/


theorem cons_dotProduct_cons (x : α) (v : Fin n → α) (y : α) (w : Fin n → α) :
                                                                          /-
                                                                            α : Type u
                                                                            n : Nat
                                                                            inst✝¹ : AddCommMonoid α
                                                                            inst✝ : Mul α
                                                                            x : α
                                                                            v : Fin n → α
                                                                            y : α
                                                                            w : Fin n → α
                                                                            ⊢ Eq (dotProduct (Matrix.vecCons x v) (Matrix.vecCons y w)) (HAdd.hAdd (HMul.h …
                                                                          -/
    dotProduct (vecCons x v) (vecCons y w) = x * y + dotProduct v w := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem col_empty (v : Fin 0 → α) : col ι v = vecEmpty :=
  empty_eq _


@[simp]
theorem col_cons (x : α) (u : Fin m → α) :
    col ι (vecCons x u) = of (vecCons (fun _ => x) (col ι u)) := by
  /-
    α : Type u
    m : Nat
    ι : Type u_1
    x : α
    u : Fin m → α
    ⊢ Eq (Matrix.col ι (Matrix.vecCons x u)) (Matrix.of (Matrix.vecCons (fun x_1 = …
  -/
  ext i j
  /-
    case a
    α : Type u
    m : Nat
    ι : Type u_1
    x : α
    u : Fin m → α
    i : Fin m.succ
    j : ι
    ⊢ Eq (Matrix.col ι (Matrix.vecCons x u) i j) (Matrix.of (Matrix.vecCons (fun x …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ i <;> simp [vecHead, vecTail]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem row_empty : row ι (vecEmpty : Fin 0 → α) = of fun _ => vecEmpty := rfl


@[simp]
theorem row_cons (x : α) (u : Fin m → α) : row ι (vecCons x u) = of fun _ => vecCons x u :=
  rfl


@[simp]
theorem transpose_empty_rows (A : Matrix m' (Fin 0) α) : Aᵀ = of ![] :=
  empty_eq _


@[simp]
theorem transpose_empty_cols (A : Matrix (Fin 0) m' α) : Aᵀ = of fun _ => ![] :=
  funext fun _ => empty_eq _


@[simp]
theorem cons_transpose (v : n' → α) (A : Matrix (Fin m) n' α) :
    (of (vecCons v A))ᵀ = of fun i => vecCons (v i) (Aᵀ i) := by
  /-
    α : Type u
    m : Nat
    n' : Type uₙ
    v : n' → α
    A : Matrix (Fin m) n' α
    ⊢ Eq (Matrix.of (Matrix.vecCons v A)).transpose (Matrix.of fun i => Matrix.vec …
  -/
  ext i j
  /-
    case a
    α : Type u
    m : Nat
    n' : Type uₙ
    v : n' → α
    A : Matrix (Fin m) n' α
    i : n'
    j : Fin m.succ
    ⊢ Eq ((Matrix.of (Matrix.vecCons v A)).transpose i j) (Matrix.of (fun i => Mat …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ j <;> simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem head_transpose (A : Matrix m' (Fin n.succ) α) :
    vecHead (of.symm Aᵀ) = vecHead ∘ of.symm A :=
  rfl


@[simp]
theorem tail_transpose (A : Matrix m' (Fin n.succ) α) : vecTail (of.symm Aᵀ) = (vecTail ∘ A)ᵀ := by
  /-
    α : Type u
    n : Nat
    m' : Type uₘ
    A : Matrix m' (Fin n.succ) α
    ⊢ Eq (Matrix.vecTail (Matrix.of.symm A.transpose)) (Matrix.transpose (Function …
  -/
  ext i j
  /-
    case h.h
    α : Type u
    n : Nat
    m' : Type uₘ
    A : Matrix m' (Fin n.succ) α
    i : Fin n
    j : m'
    ⊢ Eq (Matrix.vecTail (Matrix.of.symm A.transpose) i j) (Matrix.transpose (Func …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem empty_mul [Fintype n'] (A : Matrix (Fin 0) n' α) (B : Matrix n' o' α) : A * B = of ![] :=
  empty_eq _


@[simp]
theorem empty_mul_empty (A : Matrix m' (Fin 0) α) (B : Matrix (Fin 0) o' α) : A * B = 0 :=
  rfl


@[simp]
theorem mul_empty [Fintype n'] (A : Matrix m' n' α) (B : Matrix n' (Fin 0) α) :
    A * B = of fun _ => ![] :=
  funext fun _ => empty_eq _


theorem mul_val_succ [Fintype n'] (A : Matrix (Fin m.succ) n' α) (B : Matrix n' o' α) (i : Fin m)
    (j : o') : (A * B) i.succ j = (of (vecTail (of.symm A)) * B) i j :=
  rfl


@[simp]
theorem cons_mul [Fintype n'] (v : n' → α) (A : Fin m → n' → α) (B : Matrix n' o' α) :
    of (vecCons v A) * B = of (vecCons (v ᵥ* B) (of.symm (of A * B))) := by
  /-
    α : Type u
    m : Nat
    n' : Type uₙ
    o' : Type uₒ
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n'
    v : n' → α
    A : Fin m → n' → α
    B : Matrix n' o' α
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons v A)) B) (Matrix.of (Matrix.vecCons …
  -/
  ext i j
  /-
    case a
    α : Type u
    m : Nat
    n' : Type uₙ
    o' : Type uₒ
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n'
    v : n' → α
    A : Fin m → n' → α
    B : Matrix n' o' α
    i : Fin m.succ
    j : o'
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons v A)) B i j) (Matrix.of (Matrix.vec …
  -/
  refine Fin.cases ?_ ?_ i
    /-
      case a.refine_1
      α : Type u
      m : Nat
      n' : Type uₙ
      o' : Type uₒ
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : Fintype n'
      v : n' → α
      A : Fin m → n' → α
      B : Matrix n' o' α
      i : Fin m.succ
      j : o'
      ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons v A)) B 0 j) (Matrix.of (Matrix.vec …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case a.refine_2
    α : Type u
    m : Nat
    n' : Type uₙ
    o' : Type uₒ
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n'
    v : n' → α
    A : Fin m → n' → α
    B : Matrix n' o' α
    i : Fin m.succ
    j : o'
    ⊢ ∀ (i : Fin m), Eq (HMul.hMul (Matrix.of (Matrix.vecCons v A)) B i.succ j) (M …
  -/
  simp [mul_val_succ]
  /-
    🎉 no goals
  -/


@[simp]
theorem empty_vecMul (v : Fin 0 → α) (B : Matrix (Fin 0) o' α) : v ᵥ* B = 0 :=
  rfl


@[simp]
theorem vecMul_empty [Fintype n'] (v : n' → α) (B : Matrix n' (Fin 0) α) : v ᵥ* B = ![] :=
  empty_eq _


@[simp]
theorem cons_vecMul (x : α) (v : Fin n → α) (B : Fin n.succ → o' → α) :
    vecCons x v ᵥ* of B = x • vecHead B + v ᵥ* of (vecTail B) := by
  /-
    α : Type u
    n : Nat
    o' : Type uₒ
    inst✝ : NonUnitalNonAssocSemiring α
    x : α
    v : Fin n → α
    B : Fin n.succ → o' → α
    ⊢ Eq (Matrix.vecMul (Matrix.vecCons x v) (Matrix.of B)) (HAdd.hAdd (HSMul.hSMu …
  -/
  ext i
  /-
    case h
    α : Type u
    n : Nat
    o' : Type uₒ
    inst✝ : NonUnitalNonAssocSemiring α
    x : α
    v : Fin n → α
    B : Fin n.succ → o' → α
    i : o'
    ⊢ Eq (Matrix.vecMul (Matrix.vecCons x v) (Matrix.of B) i) (HAdd.hAdd (HSMul.hS …
  -/
  simp [vecMul]
  /-
    🎉 no goals
  -/


@[simp]
theorem vecMul_cons (v : Fin n.succ → α) (w : o' → α) (B : Fin n → o' → α) :
    v ᵥ* of (vecCons w B) = vecHead v • w + vecTail v ᵥ* of B := by
  /-
    α : Type u
    n : Nat
    o' : Type uₒ
    inst✝ : NonUnitalNonAssocSemiring α
    v : Fin n.succ → α
    w : o' → α
    B : Fin n → o' → α
    ⊢ Eq (Matrix.vecMul v (Matrix.of (Matrix.vecCons w B))) (HAdd.hAdd (HSMul.hSMu …
  -/
  ext i
  /-
    case h
    α : Type u
    n : Nat
    o' : Type uₒ
    inst✝ : NonUnitalNonAssocSemiring α
    v : Fin n.succ → α
    w : o' → α
    B : Fin n → o' → α
    i : o'
    ⊢ Eq (Matrix.vecMul v (Matrix.of (Matrix.vecCons w B)) i) (HAdd.hAdd (HSMul.hS …
  -/
  simp [vecMul]
  /-
    🎉 no goals
  -/


theorem cons_vecMul_cons (x : α) (v : Fin n → α) (w : o' → α) (B : Fin n → o' → α) :
                                                              /-
                                                                α : Type u
                                                                n : Nat
                                                                o' : Type uₒ
                                                                inst✝ : NonUnitalNonAssocSemiring α
                                                                x : α
                                                                v : Fin n → α
                                                                w : o' → α
                                                                B : Fin n → o' → α
                                                                ⊢ Eq (Matrix.vecMul (Matrix.vecCons x v) (Matrix.of (Matrix.vecCons w B))) (HA …
                                                              -/
    vecCons x v ᵥ* of (vecCons w B) = x • w + v ᵥ* of B := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem empty_mulVec [Fintype n'] (A : Matrix (Fin 0) n' α) (v : n' → α) : A *ᵥ v = ![] :=
  empty_eq _


@[simp]
theorem mulVec_empty (A : Matrix m' (Fin 0) α) (v : Fin 0 → α) : A *ᵥ v = 0 :=
  rfl


@[simp]
theorem cons_mulVec [Fintype n'] (v : n' → α) (A : Fin m → n' → α) (w : n' → α) :
    (of <| vecCons v A) *ᵥ w = vecCons (dotProduct v w) (of A *ᵥ w) := by
  /-
    α : Type u
    m : Nat
    n' : Type uₙ
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n'
    v : n' → α
    A : Fin m → n' → α
    w : n' → α
    ⊢ Eq ((Matrix.of (Matrix.vecCons v A)).mulVec w) (Matrix.vecCons (dotProduct v …
  -/
  ext i
  /-
    case h
    α : Type u
    m : Nat
    n' : Type uₙ
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n'
    v : n' → α
    A : Fin m → n' → α
    w : n' → α
    i : Fin m.succ
    ⊢ Eq ((Matrix.of (Matrix.vecCons v A)).mulVec w i) (Matrix.vecCons (dotProduct …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ i <;> simp [mulVec]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem mulVec_cons {α} [CommSemiring α] (A : m' → Fin n.succ → α) (x : α) (v : Fin n → α) :
    (of A) *ᵥ (vecCons x v) = x • vecHead ∘ A + (of (vecTail ∘ A)) *ᵥ v := by
  /-
    n : Nat
    m' : Type uₘ
    α : Type u_1
    inst✝ : CommSemiring α
    A : m' → Fin n.succ → α
    x : α
    v : Fin n → α
    ⊢ Eq ((Matrix.of A).mulVec (Matrix.vecCons x v)) (HAdd.hAdd (HSMul.hSMul x (Fu …
  -/
  ext i
  /-
    case h
    n : Nat
    m' : Type uₘ
    α : Type u_1
    inst✝ : CommSemiring α
    A : m' → Fin n.succ → α
    x : α
    v : Fin n → α
    i : m'
    ⊢ Eq ((Matrix.of A).mulVec (Matrix.vecCons x v) i) (HAdd.hAdd (HSMul.hSMul x ( …
  -/
  simp [mulVec, mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem empty_vecMulVec (v : Fin 0 → α) (w : n' → α) : vecMulVec v w = ![] :=
  empty_eq _


@[simp]
theorem vecMulVec_empty (v : m' → α) (w : Fin 0 → α) : vecMulVec v w = of fun _ => ![] :=
  funext fun _ => empty_eq _


@[simp]
theorem cons_vecMulVec (x : α) (v : Fin m → α) (w : n' → α) :
    vecMulVec (vecCons x v) w = vecCons (x • w) (vecMulVec v w) := by
  /-
    α : Type u
    m : Nat
    n' : Type uₙ
    inst✝ : NonUnitalNonAssocSemiring α
    x : α
    v : Fin m → α
    w : n' → α
    ⊢ Eq (Matrix.vecMulVec (Matrix.vecCons x v) w) (Matrix.vecCons (HSMul.hSMul x  …
  -/
  ext i
  /-
    case a
    α : Type u
    m : Nat
    n' : Type uₙ
    inst✝ : NonUnitalNonAssocSemiring α
    x : α
    v : Fin m → α
    w : n' → α
    i : Fin m.succ
    j✝ : n'
    ⊢ Eq (Matrix.vecMulVec (Matrix.vecCons x v) w i j✝) (Matrix.vecCons (HSMul.hSM …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ i <;> simp [vecMulVec]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem vecMulVec_cons (v : m' → α) (x : α) (w : Fin n → α) :
    vecMulVec v (vecCons x w) = of fun i => v i • vecCons x w := rfl


theorem smul_mat_empty {m' : Type*} (x : α) (A : Fin 0 → m' → α) : x • A = ![] :=
  empty_eq _


theorem smul_mat_cons (x : α) (v : n' → α) (A : Fin m → n' → α) :
    x • vecCons v A = vecCons (x • v) (x • A) := by
  /-
    α : Type u
    m : Nat
    n' : Type uₙ
    inst✝ : NonUnitalNonAssocSemiring α
    x : α
    v : n' → α
    A : Fin m → n' → α
    ⊢ Eq (HSMul.hSMul x (Matrix.vecCons v A)) (Matrix.vecCons (HSMul.hSMul x v) (H …
  -/
  ext i
  /-
    case h.h
    α : Type u
    m : Nat
    n' : Type uₙ
    inst✝ : NonUnitalNonAssocSemiring α
    x : α
    v : n' → α
    A : Fin m → n' → α
    i : Fin m.succ
    x✝ : n'
    ⊢ Eq (HSMul.hSMul x (Matrix.vecCons v A) i x✝) (Matrix.vecCons (HSMul.hSMul x  …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ i <;> simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem submatrix_empty (A : Matrix m' n' α) (row : Fin 0 → m') (col : o' → n') :
    submatrix A row col = ![] :=
  empty_eq _


@[simp]
theorem submatrix_cons_row (A : Matrix m' n' α) (i : m') (row : Fin m → m') (col : o' → n') :
    submatrix A (vecCons i row) col = vecCons (fun j => A i (col j)) (submatrix A row col) := by
  /-
    α : Type u
    m : Nat
    m' : Type uₘ
    n' : Type uₙ
    o' : Type uₒ
    A : Matrix m' n' α
    i : m'
    row : Fin m → m'
    col : o' → n'
    ⊢ Eq (A.submatrix (Matrix.vecCons i row) col) (Matrix.vecCons (fun j => A i (c …
  -/
  ext i j
  /-
    case a
    α : Type u
    m : Nat
    m' : Type uₘ
    n' : Type uₙ
    o' : Type uₒ
    A : Matrix m' n' α
    i✝ : m'
    row : Fin m → m'
    col : o' → n'
    i : Fin m.succ
    j : o'
    ⊢ Eq (A.submatrix (Matrix.vecCons i✝ row) col i j) (Matrix.vecCons (fun j => A …
  -/
                               /-
                                 🎉 no goals
                               -/
  refine Fin.cases ?_ ?_ i <;> simp [submatrix]
                               /-
                                 🎉 no goals
                               -/


/-- Updating a row then removing it is the same as removing it. -/
@[simp]
theorem submatrix_updateRow_succAbove (A : Matrix (Fin m.succ) n' α) (v : n' → α) (f : o' → n')
    (i : Fin m.succ) : (A.updateRow i v).submatrix i.succAbove f = A.submatrix i.succAbove f :=
  ext fun r s => (congr_fun (updateRow_ne (Fin.succAbove_ne i r) : _ = A _) (f s) : _)


/-- Updating a column then removing it is the same as removing it. -/
@[simp]
theorem submatrix_updateCol_succAbove (A : Matrix m' (Fin n.succ) α) (v : m' → α) (f : o' → m')
    (i : Fin n.succ) : (A.updateCol i v).submatrix f i.succAbove = A.submatrix f i.succAbove :=
  ext fun _r s => updateCol_ne (Fin.succAbove_ne i s)


@[deprecated (since := "2024-12-11")]
alias submatrix_updateColumn_succAbove := submatrix_updateCol_succAbove


theorem one_fin_two : (1 : Matrix (Fin 2) (Fin 2) α) = !![1, 0; 0, 1] := by
  /-
    α : Type u
    inst✝¹ : Zero α
    inst✝ : One α
    ⊢ Eq 1 (Matrix.of (Matrix.vecCons (Matrix.vecCons 1 (Matrix.vecCons 0 Matrix.v …
  -/
  ext i j
  /-
    case a
    α : Type u
    inst✝¹ : Zero α
    inst✝ : One α
    i j : Fin 2
    ⊢ Eq (1 i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons 1 (Matrix.vecCons 0 Ma …
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
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem one_fin_three : (1 : Matrix (Fin 3) (Fin 3) α) = !![1, 0, 0; 0, 1, 0; 0, 0, 1] := by
  /-
    α : Type u
    inst✝¹ : Zero α
    inst✝ : One α
    ⊢ Eq 1 (Matrix.of (Matrix.vecCons (Matrix.vecCons 1 (Matrix.vecCons 0 (Matrix. …
  -/
  ext i j
  /-
    case a
    α : Type u
    inst✝¹ : Zero α
    inst✝ : One α
    i j : Fin 3
    ⊢ Eq (1 i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons 1 (Matrix.vecCons 0 (M …
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
                                  /-
                                    🎉 no goals
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
                                  /-
                                    🎉 no goals
                                  -/
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem natCast_fin_two (n : ℕ) : (n : Matrix (Fin 2) (Fin 2) α) = !![↑n, 0; 0, ↑n] := by
  /-
    α : Type u
    inst✝ : AddMonoidWithOne α
    n : Nat
    ⊢ Eq (↑n) (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑n) (Matrix.vecCons 0 Ma …
  -/
  ext i j
  /-
    case a
    α : Type u
    inst✝ : AddMonoidWithOne α
    n : Nat
    i j : Fin 2
    ⊢ Eq (↑n i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑n) (Matrix.vecCons  …
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
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem natCast_fin_three (n : ℕ) :
    (n : Matrix (Fin 3) (Fin 3) α) = !![↑n, 0, 0; 0, ↑n, 0; 0, 0, ↑n] := by
  /-
    α : Type u
    inst✝ : AddMonoidWithOne α
    n : Nat
    ⊢ Eq (↑n) (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑n) (Matrix.vecCons 0 (M …
  -/
  ext i j
  /-
    case a
    α : Type u
    inst✝ : AddMonoidWithOne α
    n : Nat
    i j : Fin 3
    ⊢ Eq (↑n i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑n) (Matrix.vecCons  …
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
                                  /-
                                    🎉 no goals
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
                                  /-
                                    🎉 no goals
                                  -/
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/

-- See note [no_index around OfNat.ofNat]

theorem ofNat_fin_two (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n) : Matrix (Fin 2) (Fin 2) α) =
      !![OfNat.ofNat n, 0; 0, OfNat.ofNat n] :=
  natCast_fin_two _

-- See note [no_index around OfNat.ofNat]

theorem ofNat_fin_three (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n) : Matrix (Fin 3) (Fin 3) α) =
      !![OfNat.ofNat n, 0, 0; 0, OfNat.ofNat n, 0; 0, 0, OfNat.ofNat n] :=
  natCast_fin_three _


theorem eta_fin_two (A : Matrix (Fin 2) (Fin 2) α) : A = !![A 0 0, A 0 1; A 1 0, A 1 1] := by
  /-
    α : Type u
    A : Matrix (Fin 2) (Fin 2) α
    ⊢ Eq A (Matrix.of (Matrix.vecCons (Matrix.vecCons (A 0 0) (Matrix.vecCons (A 0 …
  -/
  ext i j
  /-
    case a
    α : Type u
    A : Matrix (Fin 2) (Fin 2) α
    i j : Fin 2
    ⊢ Eq (A i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons (A 0 0) (Matrix.vecCon …
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
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem eta_fin_three (A : Matrix (Fin 3) (Fin 3) α) :
    A = !![A 0 0, A 0 1, A 0 2;
           A 1 0, A 1 1, A 1 2;
           A 2 0, A 2 1, A 2 2] := by
  /-
    α : Type u
    A : Matrix (Fin 3) (Fin 3) α
    ⊢ Eq A (Matrix.of (Matrix.vecCons (Matrix.vecCons (A 0 0) (Matrix.vecCons (A 0 …
  -/
  ext i j
  /-
    case a
    α : Type u
    A : Matrix (Fin 3) (Fin 3) α
    i j : Fin 3
    ⊢ Eq (A i j) (Matrix.of (Matrix.vecCons (Matrix.vecCons (A 0 0) (Matrix.vecCon …
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
                                  /-
                                    🎉 no goals
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
                                  /-
                                    🎉 no goals
                                  -/
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem mul_fin_two [AddCommMonoid α] [Mul α] (a₁₁ a₁₂ a₂₁ a₂₂ b₁₁ b₁₂ b₂₁ b₂₂ : α) :
    !![a₁₁, a₁₂;
       a₂₁, a₂₂] * !![b₁₁, b₁₂;
                      b₂₁, b₂₂] = !![a₁₁ * b₁₁ + a₁₂ * b₂₁, a₁₁ * b₁₂ + a₁₂ * b₂₂;
                                     a₂₁ * b₁₁ + a₂₂ * b₂₁, a₂₁ * b₁₂ + a₂₂ * b₂₂] := by
  /-
    α : Type u
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    a₁₁ a₁₂ a₂₁ a₂₂ b₁₁ b₁₂ b₂₁ b₂₂ : α
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons a₁₁ (Matrix.vecCons …
  -/
  ext i j
  /-
    case a
    α : Type u
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    a₁₁ a₁₂ a₂₁ a₂₂ b₁₁ b₁₂ b₂₁ b₂₂ : α
    i j : Fin 2
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons a₁₁ (Matrix.vecCons …
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
  fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, dotProduct, Fin.sum_univ_succ]
                                  /-
                                    🎉 no goals
                                  -/


theorem mul_fin_three [AddCommMonoid α] [Mul α]
    (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ b₁₁ b₁₂ b₁₃ b₂₁ b₂₂ b₂₃ b₃₁ b₃₂ b₃₃ : α) :
    !![a₁₁, a₁₂, a₁₃;
       a₂₁, a₂₂, a₂₃;
       a₃₁, a₃₂, a₃₃] * !![b₁₁, b₁₂, b₁₃;
                           b₂₁, b₂₂, b₂₃;
                           b₃₁, b₃₂, b₃₃] =
    !![a₁₁*b₁₁ + a₁₂*b₂₁ + a₁₃*b₃₁, a₁₁*b₁₂ + a₁₂*b₂₂ + a₁₃*b₃₂, a₁₁*b₁₃ + a₁₂*b₂₃ + a₁₃*b₃₃;
       a₂₁*b₁₁ + a₂₂*b₂₁ + a₂₃*b₃₁, a₂₁*b₁₂ + a₂₂*b₂₂ + a₂₃*b₃₂, a₂₁*b₁₃ + a₂₂*b₂₃ + a₂₃*b₃₃;
       a₃₁*b₁₁ + a₃₂*b₂₁ + a₃₃*b₃₁, a₃₁*b₁₂ + a₃₂*b₂₂ + a₃₃*b₃₂, a₃₁*b₁₃ + a₃₂*b₂₃ + a₃₃*b₃₃] := by
  /-
    α : Type u
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ b₁₁ b₁₂ b₁₃ b₂₁ b₂₂ b₂₃ b₃₁ b₃₂ b₃₃ : α
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons a₁₁ (Matrix.vecCons …
  -/
  ext i j
  /-
    case a
    α : Type u
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ b₁₁ b₁₂ b₁₃ b₂₁ b₂₂ b₂₃ b₃₁ b₃₂ b₃₃ : α
    i j : Fin 3
    ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons a₁₁ (Matrix.vecCons …
  -/
  fin_cases i <;> fin_cases j
        /-
          case a.«_@».Mathlib.Data.Matrix.Defs._hyg.177.«0».«0»
          α : Type u
          inst✝¹ : AddCommMonoid α
          inst✝ : Mul α
          a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ b₁₁ b₁₂ b₁₃ b₂₁ b₂₂ b₂₃ b₃₁ b₃₂ b₃₃ : α
          ⊢ Eq (HMul.hMul (Matrix.of (Matrix.vecCons (Matrix.vecCons a₁₁ (Matrix.vecCons …
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
        /-
          🎉 no goals
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
        /-
          🎉 no goals
        -/
    <;> simp [Matrix.mul_apply, dotProduct, Fin.sum_univ_succ, ← add_assoc]
        /-
          🎉 no goals
        -/


theorem vec2_eq {a₀ a₁ b₀ b₁ : α} (h₀ : a₀ = b₀) (h₁ : a₁ = b₁) : ![a₀, a₁] = ![b₀, b₁] := by
  /-
    α : Type u
    a₀ a₁ b₀ b₁ : α
    h₀ : Eq a₀ b₀
    h₁ : Eq a₁ b₁
    ⊢ Eq (Matrix.vecCons a₀ (Matrix.vecCons a₁ Matrix.vecEmpty)) (Matrix.vecCons b …
  -/
  subst_vars
  /-
    α : Type u
    b₀ b₁ : α
    ⊢ Eq (Matrix.vecCons b₀ (Matrix.vecCons b₁ Matrix.vecEmpty)) (Matrix.vecCons b …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem vec3_eq {a₀ a₁ a₂ b₀ b₁ b₂ : α} (h₀ : a₀ = b₀) (h₁ : a₁ = b₁) (h₂ : a₂ = b₂) :
    ![a₀, a₁, a₂] = ![b₀, b₁, b₂] := by
  /-
    α : Type u
    a₀ a₁ a₂ b₀ b₁ b₂ : α
    h₀ : Eq a₀ b₀
    h₁ : Eq a₁ b₁
    h₂ : Eq a₂ b₂
    ⊢ Eq (Matrix.vecCons a₀ (Matrix.vecCons a₁ (Matrix.vecCons a₂ Matrix.vecEmpty) …
  -/
  subst_vars
  /-
    α : Type u
    b₀ b₁ b₂ : α
    ⊢ Eq (Matrix.vecCons b₀ (Matrix.vecCons b₁ (Matrix.vecCons b₂ Matrix.vecEmpty) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem vec2_add [Add α] (a₀ a₁ b₀ b₁ : α) : ![a₀, a₁] + ![b₀, b₁] = ![a₀ + b₀, a₁ + b₁] := by
  /-
    α : Type u
    inst✝ : Add α
    a₀ a₁ b₀ b₁ : α
    ⊢ Eq (HAdd.hAdd (Matrix.vecCons a₀ (Matrix.vecCons a₁ Matrix.vecEmpty)) (Matri …
  -/
  rw [cons_add_cons, cons_add_cons, empty_add_empty]
  /-
    🎉 no goals
  -/


theorem vec3_add [Add α] (a₀ a₁ a₂ b₀ b₁ b₂ : α) :
    ![a₀, a₁, a₂] + ![b₀, b₁, b₂] = ![a₀ + b₀, a₁ + b₁, a₂ + b₂] := by
  /-
    α : Type u
    inst✝ : Add α
    a₀ a₁ a₂ b₀ b₁ b₂ : α
    ⊢ Eq (HAdd.hAdd (Matrix.vecCons a₀ (Matrix.vecCons a₁ (Matrix.vecCons a₂ Matri …
  -/
  rw [cons_add_cons, cons_add_cons, cons_add_cons, empty_add_empty]
  /-
    🎉 no goals
  -/


theorem smul_vec2 {R : Type*} [SMul R α] (x : R) (a₀ a₁ : α) :
                                            /-
                                              α : Type u
                                              R : Type u_1
                                              inst✝ : SMul R α
                                              x : R
                                              a₀ a₁ : α
                                              ⊢ Eq (HSMul.hSMul x (Matrix.vecCons a₀ (Matrix.vecCons a₁ Matrix.vecEmpty))) ( …
                                            -/
    x • ![a₀, a₁] = ![x • a₀, x • a₁] := by rw [smul_cons, smul_cons, smul_empty]
                                            /-
                                              🎉 no goals
                                            -/


theorem smul_vec3 {R : Type*} [SMul R α] (x : R) (a₀ a₁ a₂ : α) :
    x • ![a₀, a₁, a₂] = ![x • a₀, x • a₁, x • a₂] := by
  /-
    α : Type u
    R : Type u_1
    inst✝ : SMul R α
    x : R
    a₀ a₁ a₂ : α
    ⊢ Eq (HSMul.hSMul x (Matrix.vecCons a₀ (Matrix.vecCons a₁ (Matrix.vecCons a₂ M …
  -/
  rw [smul_cons, smul_cons, smul_cons, smul_empty]
  /-
    🎉 no goals
  -/


theorem vec2_dotProduct' {a₀ a₁ b₀ b₁ : α} : ![a₀, a₁] ⬝ᵥ ![b₀, b₁] = a₀ * b₀ + a₁ * b₁ := by
  /-
    α : Type u
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    a₀ a₁ b₀ b₁ : α
    ⊢ Eq (dotProduct (Matrix.vecCons a₀ (Matrix.vecCons a₁ Matrix.vecEmpty)) (Matr …
  -/
  rw [cons_dotProduct_cons, cons_dotProduct_cons, dotProduct_empty, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem vec2_dotProduct (v w : Fin 2 → α) : v ⬝ᵥ w = v 0 * w 0 + v 1 * w 1 :=
  vec2_dotProduct'


theorem vec3_dotProduct' {a₀ a₁ a₂ b₀ b₁ b₂ : α} :
    ![a₀, a₁, a₂] ⬝ᵥ ![b₀, b₁, b₂] = a₀ * b₀ + a₁ * b₁ + a₂ * b₂ := by
  rw [cons_dotProduct_cons, cons_dotProduct_cons, cons_dotProduct_cons, dotProduct_empty,
    add_zero, add_assoc]


@[simp]
theorem vec3_dotProduct (v w : Fin 3 → α) : v ⬝ᵥ w = v 0 * w 0 + v 1 * w 1 + v 2 * w 2 :=
  vec3_dotProduct'


