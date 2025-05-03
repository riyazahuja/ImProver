/-- An explicit form for the constant in the triangle removal lemma.

Note that this depends on `SzemerediRegularity.bound`, which is a tower-type exponential. This means
`triangleRemovalBound` is in practice absolutely tiny. -/
noncomputable def triangleRemovalBound (ε : ℝ) : ℝ :=
  min (2 * ⌈4/ε⌉₊^3)⁻¹ ((1 - ε/4) * (ε/(16 * bound (ε/8) ⌈4/ε⌉₊))^3)


lemma triangleRemovalBound_pos (hε : 0 < ε) (hε₁ : ε ≤ 1) : 0 < triangleRemovalBound ε := by
  /-
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    ⊢ LT.lt 0 (SimpleGraph.triangleRemovalBound ε)
  -/
  have : 0 < 1 - ε / 4 := by linarith
  /-
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    this : LT.lt 0 (HSub.hSub 1 (HDiv.hDiv ε 4))
    ⊢ LT.lt 0 (SimpleGraph.triangleRemovalBound ε)
  -/
  unfold triangleRemovalBound
  /-
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    this : LT.lt 0 (HSub.hSub 1 (HDiv.hDiv ε 4))
    ⊢ LT.lt 0 (Min.min (Inv.inv (HMul.hMul 2 (HPow.hPow (↑(Nat.ceil (HDiv.hDiv 4 ε …
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma triangleRemovalBound_nonpos (hε : ε ≤ 0) : triangleRemovalBound ε ≤ 0 := by
  /-
    ε : Real
    hε : LE.le ε 0
    ⊢ LE.le (SimpleGraph.triangleRemovalBound ε) 0
  -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
  rw [triangleRemovalBound, ceil_eq_zero.2 (div_nonpos_of_nonneg_of_nonpos _ hε)] <;> simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


lemma triangleRemovalBound_mul_cube_lt (hε : 0 < ε) :
    triangleRemovalBound ε * ⌈4 / ε⌉₊ ^ 3 < 1 := by
  calc
    _ ≤ (2 * ⌈4 / ε⌉₊ ^ 3 : ℝ)⁻¹ * ↑⌈4 / ε⌉₊ ^ 3 := by gcongr; exact min_le_left _ _
    _ = 2⁻¹ := by rw [mul_inv, inv_mul_cancel_right₀]; positivity
    _ < 1 := by norm_num


private lemma aux {n k : ℕ} (hk : 0 < k) (hn : k ≤ n) : n < 2 * k * (n / k) := by
  rw [mul_assoc, two_mul, ← add_lt_add_iff_right (n % k), add_right_comm, add_assoc,
    mod_add_div n k, add_comm, add_lt_add_iff_right]
  /-
    n k : Nat
    hk : LT.lt 0 k
    hn : LE.le k n
    ⊢ LT.lt (HMod.hMod n k) (HMul.hMul k (HDiv.hDiv n k))
  -/
  apply (mod_lt n hk).trans_le
  /-
    n k : Nat
    hk : LT.lt 0 k
    hn : LE.le k n
    ⊢ LE.le k (HMul.hMul k (HDiv.hDiv n k))
  -/
  simpa using Nat.mul_le_mul_left k ((Nat.one_le_div_iff hk).2 hn)
  /-
    🎉 no goals
  -/


private lemma card_bound (hP₁ : P.IsEquipartition) (hP₃ : #P.parts ≤ bound (ε / 8) ⌈4/ε⌉₊)
    (hX : s ∈ P.parts) : card α / (2 * bound (ε / 8) ⌈4 / ε⌉₊ : ℝ) ≤ #s := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Finset α
    P : Finpartition Finset.univ
    ε : Real
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    hX : Membership.mem P.parts s
    ⊢ LE.le (HDiv.hDiv (↑(Fintype.card α)) (HMul.hMul 2 ↑(SzemerediRegularity.boun …
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      s : Finset α
      P : Finpartition Finset.univ
      ε : Real
      hP₁ : P.IsEquipartition
      hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
      hX : Membership.mem P.parts s
      h✝ : IsEmpty α
      ⊢ LE.le (HDiv.hDiv (↑(Fintype.card α)) (HMul.hMul 2 ↑(SzemerediRegularity.boun …
    -/
  · simp [Fintype.card_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    s : Finset α
    P : Finpartition Finset.univ
    ε : Real
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    hX : Membership.mem P.parts s
    h✝ : Nonempty α
    ⊢ LE.le (HDiv.hDiv (↑(Fintype.card α)) (HMul.hMul 2 ↑(SzemerediRegularity.boun …
  -/
  have := Finset.Nonempty.card_pos ⟨_, hX⟩
  calc
    _ ≤ card α / (2 * #P.parts : ℝ) := by gcongr
    _ ≤ ↑(card α / #P.parts) :=
      (div_le_iff₀' (by positivity)).2 <| mod_cast (aux ‹_› P.card_parts_le_card).le
    _ ≤ (#s : ℝ) := mod_cast hP₁.average_le_card_part hX


private lemma triangle_removal_aux (hε : 0 < ε) (hP₁ : P.IsEquipartition)
    (hP₃ : #P.parts ≤ bound (ε / 8) ⌈4/ε⌉₊)
    (ht : t ∈ (G.regularityReduced P (ε/8) (ε/4)).cliqueFinset 3) :
    triangleRemovalBound ε * card α ^ 3 ≤ #(G.cliqueFinset 3) := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    t : Finset α
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    ht : Membership.mem ((SimpleGraph.regularityReduced P G (HDiv.hDiv ε 8) (HDiv. …
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  rw [mem_cliqueFinset_iff, is3Clique_iff] at ht
  obtain ⟨x, y, z, ⟨-, s, hX, Y, hY, xX, yY, nXY, uXY, dXY⟩,
                   ⟨-, X', hX', Z, hZ, xX', zZ, nXZ, uXZ, dXZ⟩,
                   ⟨-, Y', hY', Z', hZ', yY', zZ', nYZ, uYZ, dYZ⟩, rfl⟩ := ht
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    X' : Finset α
    hX' : Membership.mem P.parts X'
    Z : Finset α
    hZ : Membership.mem P.parts Z
    xX' : Membership.mem X' x
    zZ : Membership.mem Z z
    nXZ : Ne X' Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) X' Z
    dXZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity X' Z)
    Y' : Finset α
    hY' : Membership.mem P.parts Y'
    Z' : Finset α
    hZ' : Membership.mem P.parts Z'
    yY' : Membership.mem Y' y
    zZ' : Membership.mem Z' z
    nYZ : Ne Y' Z'
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y' Z'
    dYZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y' Z')
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  cases P.disjoint.elim hX hX' (not_disjoint_iff.2 ⟨x, xX, xX'⟩)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    Z : Finset α
    hZ : Membership.mem P.parts Z
    zZ : Membership.mem Z z
    Y' : Finset α
    hY' : Membership.mem P.parts Y'
    Z' : Finset α
    hZ' : Membership.mem P.parts Z'
    yY' : Membership.mem Y' y
    zZ' : Membership.mem Z' z
    nYZ : Ne Y' Z'
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y' Z'
    dYZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y' Z')
    hX' : Membership.mem P.parts s
    xX' : Membership.mem s x
    nXZ : Ne s Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) s Z
    dXZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Z)
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  cases P.disjoint.elim hY hY' (not_disjoint_iff.2 ⟨y, yY, yY'⟩)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    Z : Finset α
    hZ : Membership.mem P.parts Z
    zZ : Membership.mem Z z
    Z' : Finset α
    hZ' : Membership.mem P.parts Z'
    zZ' : Membership.mem Z' z
    hX' : Membership.mem P.parts s
    xX' : Membership.mem s x
    nXZ : Ne s Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) s Z
    dXZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Z)
    hY' : Membership.mem P.parts Y
    yY' : Membership.mem Y y
    nYZ : Ne Y Z'
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y Z'
    dYZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y Z')
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  cases P.disjoint.elim hZ hZ' (not_disjoint_iff.2 ⟨z, zZ, zZ'⟩)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    Z : Finset α
    hZ : Membership.mem P.parts Z
    zZ : Membership.mem Z z
    hX' : Membership.mem P.parts s
    xX' : Membership.mem s x
    nXZ : Ne s Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) s Z
    dXZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Z)
    hY' : Membership.mem P.parts Y
    yY' : Membership.mem Y y
    hZ' : Membership.mem P.parts Z
    zZ' : Membership.mem Z z
    nYZ : Ne Y Z
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y Z
    dYZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y Z)
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have dXY := P.disjoint hX hY nXY
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY✝ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    Z : Finset α
    hZ : Membership.mem P.parts Z
    zZ : Membership.mem Z z
    hX' : Membership.mem P.parts s
    xX' : Membership.mem s x
    nXZ : Ne s Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) s Z
    dXZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Z)
    hY' : Membership.mem P.parts Y
    yY' : Membership.mem Y y
    hZ' : Membership.mem P.parts Z
    zZ' : Membership.mem Z z
    nYZ : Ne Y Z
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y Z
    dYZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y Z)
    dXY : Function.onFun Disjoint id s Y
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have dXZ := P.disjoint hX hZ nXZ
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY✝ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    Z : Finset α
    hZ : Membership.mem P.parts Z
    zZ : Membership.mem Z z
    hX' : Membership.mem P.parts s
    xX' : Membership.mem s x
    nXZ : Ne s Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) s Z
    dXZ✝ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Z)
    hY' : Membership.mem P.parts Y
    yY' : Membership.mem Y y
    hZ' : Membership.mem P.parts Z
    zZ' : Membership.mem Z z
    nYZ : Ne Y Z
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y Z
    dYZ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y Z)
    dXY : Function.onFun Disjoint id s Y
    dXZ : Function.onFun Disjoint id s Z
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have dYZ := P.disjoint hY hZ nYZ
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    hε : LT.lt 0 ε
    hP₁ : P.IsEquipartition
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) (Nat.ceil  …
    x y z : α
    s : Finset α
    hX : Membership.mem P.parts s
    Y : Finset α
    hY : Membership.mem P.parts Y
    xX : Membership.mem s x
    yY : Membership.mem Y y
    nXY : Ne s Y
    uXY : G.IsUniform (HDiv.hDiv ε 8) s Y
    dXY✝ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Y)
    Z : Finset α
    hZ : Membership.mem P.parts Z
    zZ : Membership.mem Z z
    hX' : Membership.mem P.parts s
    xX' : Membership.mem s x
    nXZ : Ne s Z
    uXZ : G.IsUniform (HDiv.hDiv ε 8) s Z
    dXZ✝ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity s Z)
    hY' : Membership.mem P.parts Y
    yY' : Membership.mem Y y
    hZ' : Membership.mem P.parts Z
    zZ' : Membership.mem Z z
    nYZ : Ne Y Z
    uYZ : G.IsUniform (HDiv.hDiv ε 8) Y Z
    dYZ✝ : LE.le (HDiv.hDiv ε 4) ↑(G.edgeDensity Y Z)
    dXY : Function.onFun Disjoint id s Y
    dXZ : Function.onFun Disjoint id s Z
    dYZ : Function.onFun Disjoint id Y Z
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have that : 2 * (ε/8) = ε/4 := by ring
  have : 0 ≤ 1 - 2 * (ε / 8) := by
    have : ε / 4 ≤ 1 := ‹ε / 4 ≤ _›.trans (by exact mod_cast G.edgeDensity_le_one _ _); linarith
  calc
    _ ≤ (1 - ε/4) * (ε/(16 * bound (ε/8) ⌈4/ε⌉₊))^3 * card α ^ 3 := by
      gcongr; exact min_le_right _ _
    _ = (1 - 2 * (ε / 8)) * (ε / 8) ^ 3 * (card α / (2 * bound (ε / 8) ⌈4 / ε⌉₊)) *
          (card α / (2 * bound (ε / 8) ⌈4 / ε⌉₊)) * (card α / (2 * bound (ε / 8) ⌈4 / ε⌉₊)) := by
      ring
    _ ≤ (1 - 2 * (ε / 8)) * (ε / 8) ^ 3 * #s * #Y * #Z := by
      gcongr <;> exact card_bound hP₁ hP₃ ‹_›
    _ ≤ _ :=
      triangle_counting G (by rwa [that]) uXY dXY (by rwa [that]) uXZ dXZ (by rwa [that]) uYZ dYZ


lemma regularityReduced_edges_card_aux [Nonempty α] (hε : 0 < ε) (hP : P.IsEquipartition)
    (hPε : P.IsUniform G (ε/8)) (hP' : 4 / ε ≤ #P.parts) :
    2 * (#G.edgeFinset - #(G.regularityReduced P (ε/8) (ε/4)).edgeFinset : ℝ)
      < 2 * ε * (card α ^ 2 : ℕ) := by
  /-
    α : Type u_1
    inst✝³ : DecidableEq α
    inst✝² : Fintype α
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hPε : P.IsUniform G (HDiv.hDiv ε 8)
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    ⊢ LT.lt (HMul.hMul 2 (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityRed …
  -/
  let A := (P.nonUniforms G (ε/8)).biUnion fun (U, V) ↦ U ×ˢ V
  /-
    α : Type u_1
    inst✝³ : DecidableEq α
    inst✝² : Fintype α
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hPε : P.IsUniform G (HDiv.hDiv ε 8)
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    A : Finset (Prod α α) := (P.nonUniforms G (HDiv.hDiv ε 8)).biUnion fun x => Si …
    ⊢ LT.lt (HMul.hMul 2 (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityRed …
  -/
  let B := P.parts.biUnion offDiag
  /-
    α : Type u_1
    inst✝³ : DecidableEq α
    inst✝² : Fintype α
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    P : Finpartition Finset.univ
    ε : Real
    inst✝ : Nonempty α
    hε : LT.lt 0 ε
    hP : P.IsEquipartition
    hPε : P.IsUniform G (HDiv.hDiv ε 8)
    hP' : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    A : Finset (Prod α α) := (P.nonUniforms G (HDiv.hDiv ε 8)).biUnion fun x => Si …
    B : Finset (Prod α α) := P.parts.biUnion Finset.offDiag
    ⊢ LT.lt (HMul.hMul 2 (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityRed …
  -/
  let C := (P.sparsePairs G (ε/4)).biUnion fun (U, V) ↦ G.interedges U V
  calc
    _ = (#((univ ×ˢ univ).filter fun (x, y) ↦
          G.Adj x y ∧ ¬(G.regularityReduced P (ε / 8) (ε /4)).Adj x y) : ℝ) := by
      rw [univ_product_univ, mul_sub, filter_and_not, cast_card_sdiff]
      · norm_cast
        rw [two_mul_card_edgeFinset, two_mul_card_edgeFinset]
      · exact monotone_filter_right _ fun xy hxy ↦ regularityReduced_le hxy
    _ ≤ #(A ∪ B ∪ C) := by gcongr; exact unreduced_edges_subset
    _ ≤ #(A ∪ B) + #C := mod_cast (card_union_le _ _)
    _ ≤ #A + #B + #C := by gcongr; exact mod_cast card_union_le _ _
    _ < 4 * (ε / 8) * card α ^ 2 + _ + _ := by
      gcongr; exact hP.sum_nonUniforms_lt univ_nonempty (by positivity) hPε
    _ ≤ _ + ε / 2 * card α ^ 2 + 4 * (ε / 4) * card α ^ 2 := by
      gcongr
      · exact hP.card_biUnion_offDiag_le hε hP'
      · exact hP.card_interedges_sparsePairs_le (G := G) (ε := ε / 4) (by positivity)
    _ = 2 * ε * (card α ^ 2 : ℕ) := by norm_cast; ring


/-- **Triangle Removal Lemma**. If not all triangles can be removed by removing few edges (on the
order of `(card α)^2`), then there were many triangles to start with (on the order of
`(card α)^3`). -/
lemma FarFromTriangleFree.le_card_cliqueFinset (hG : G.FarFromTriangleFree ε) :
    triangleRemovalBound ε * card α ^ 3 ≤ #(G.cliqueFinset 3) := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      hG : G.FarFromTriangleFree ε
      h✝ : IsEmpty α
      ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
    -/
  · simp [Fintype.card_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  obtain hε | hε := le_or_lt ε 0
    /-
      case inr.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      G : SimpleGraph α
      inst✝ : DecidableRel G.Adj
      ε : Real
      hG : G.FarFromTriangleFree ε
      h✝ : Nonempty α
      hε : LE.le ε 0
      ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
    -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  · apply (mul_nonpos_of_nonpos_of_nonneg (triangleRemovalBound_nonpos hε) _).trans <;> positivity
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  /-
    case inr.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  let l : ℕ := ⌈4 / ε⌉₊
  /-
    case inr.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have hl : 4/ε ≤ l := le_ceil (4/ε)
  /-
    case inr.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  cases' le_total (card α) l with hl' hl'
  · calc
      _ ≤ triangleRemovalBound ε * ↑l ^ 3 := by
        gcongr; exact (triangleRemovalBound_pos hε hG.lt_one.le).le
      _ ≤ (1 : ℝ) := (triangleRemovalBound_mul_cube_lt hε).le
      _ ≤ _ := by simpa [one_le_iff_ne_zero] using (hG.cliqueFinset_nonempty hε).card_pos.ne'
  /-
    case inr.inr.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  obtain ⟨P, hP₁, hP₂, hP₃, hP₄⟩ := szemeredi_regularity G (by positivity : 0 < ε / 8) hl'
  /-
    case inr.inr.inr.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le l P.parts.card
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) l)
    hP₄ : P.IsUniform G (HDiv.hDiv ε 8)
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have : 4/ε ≤ #P.parts := hl.trans (cast_le.2 hP₂)
  /-
    case inr.inr.inr.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le l P.parts.card
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) l)
    hP₄ : P.IsUniform G (HDiv.hDiv ε 8)
    this : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  have k := regularityReduced_edges_card_aux hε hP₁ hP₄ this
  /-
    case inr.inr.inr.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le l P.parts.card
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) l)
    hP₄ : P.IsUniform G (HDiv.hDiv ε 8)
    this : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    k : LT.lt (HMul.hMul 2 (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityR …
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  rw [mul_assoc] at k
  /-
    case inr.inr.inr.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le l P.parts.card
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) l)
    hP₄ : P.IsUniform G (HDiv.hDiv ε 8)
    this : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    k : LT.lt (HMul.hMul 2 (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityR …
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  replace k := lt_of_mul_lt_mul_left k zero_le_two
  /-
    case inr.inr.inr.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le l P.parts.card
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) l)
    hP₄ : P.IsUniform G (HDiv.hDiv ε 8)
    this : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    k : LT.lt (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityReduced P G (H …
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  obtain ⟨t, ht⟩ := hG.cliqueFinset_nonempty' regularityReduced_le k
  /-
    case inr.inr.inr.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : G.FarFromTriangleFree ε
    h✝ : Nonempty α
    hε : LT.lt 0 ε
    l : Nat := Nat.ceil (HDiv.hDiv 4 ε)
    hl : LE.le (HDiv.hDiv 4 ε) ↑l
    hl' : LE.le l (Fintype.card α)
    P : Finpartition Finset.univ
    hP₁ : P.IsEquipartition
    hP₂ : LE.le l P.parts.card
    hP₃ : LE.le P.parts.card (SzemerediRegularity.bound (HDiv.hDiv ε 8) l)
    hP₄ : P.IsUniform G (HDiv.hDiv ε 8)
    this : LE.le (HDiv.hDiv 4 ε) ↑P.parts.card
    k : LT.lt (HSub.hSub ↑G.edgeFinset.card ↑(SimpleGraph.regularityReduced P G (H …
    t : Finset α
    ht : Membership.mem ((SimpleGraph.regularityReduced P G (HDiv.hDiv ε 8) (HDiv. …
    ⊢ LE.le (HMul.hMul (SimpleGraph.triangleRemovalBound ε) (HPow.hPow (↑(Fintype. …
  -/
  exact triangle_removal_aux hε hP₁ hP₃ ht
  /-
    🎉 no goals
  -/


/-- **Triangle Removal Lemma**. If there are not too many triangles (on the order of `(card α)^3`)
then they can all be removed by removing a few edges (on the order of `(card α)^2`). -/
lemma triangle_removal (hG : #(G.cliqueFinset 3) < triangleRemovalBound ε * card α ^ 3) :
    ∃ G' ≤ G, ∃ _ : DecidableRel G'.Adj,
      (#G.edgeFinset - #G'.edgeFinset : ℝ) < ε * (card α^2 : ℕ) ∧ G'.CliqueFree 3 := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : LT.lt (↑(G.cliqueFinset 3).card) (HMul.hMul (SimpleGraph.triangleRemovalB …
    ⊢ Exists fun G' => And (LE.le G' G) (Exists fun x => And (LT.lt (HSub.hSub ↑G. …
  -/
  by_contra! h
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : LT.lt (↑(G.cliqueFinset 3).card) (HMul.hMul (SimpleGraph.triangleRemovalB …
    h : ∀ (G' : SimpleGraph α), LE.le G' G → ∀ (x : DecidableRel G'.Adj), LT.lt (H …
    ⊢ False
  -/
  refine hG.not_le (farFromTriangleFree_iff.2 ?_).le_card_cliqueFinset
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    G : SimpleGraph α
    inst✝ : DecidableRel G.Adj
    ε : Real
    hG : LT.lt (↑(G.cliqueFinset 3).card) (HMul.hMul (SimpleGraph.triangleRemovalB …
    h : ∀ (G' : SimpleGraph α), LE.le G' G → ∀ (x : DecidableRel G'.Adj), LT.lt (H …
    ⊢ ∀ ⦃H : SimpleGraph α⦄ [inst : DecidableRel H.Adj], LE.le H G → H.CliqueFree  …
  -/
  intros G' _ hG hG'
  /-
    α : Type u_1
    inst✝³ : DecidableEq α
    inst✝² : Fintype α
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    ε : Real
    hG✝ : LT.lt (↑(G.cliqueFinset 3).card) (HMul.hMul (SimpleGraph.triangleRemoval …
    h : ∀ (G' : SimpleGraph α), LE.le G' G → ∀ (x : DecidableRel G'.Adj), LT.lt (H …
    G' : SimpleGraph α
    inst✝ : DecidableRel G'.Adj
    hG : LE.le G' G
    hG' : G'.CliqueFree 3
    ⊢ LE.le (HMul.hMul ε ↑(HPow.hPow (Fintype.card α) 2)) (HSub.hSub ↑G.edgeFinset …
  -/
  exact le_of_not_lt fun i ↦ h G' hG _ i hG'
  /-
    🎉 no goals
  -/


/-- Extension for the `positivity` tactic: `SimpleGraph.triangleRemovalBound ε` is positive if
`0 < ε ≤ 1`.

Note this looks for `ε ≤ 1` in the context. -/
@[positivity triangleRemovalBound _]
def evalTriangleRemovalBound : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(triangleRemovalBound $ε) =>
    let some hε₁ ← findLocalDeclWithTypeQ? q($ε ≤ 1) | failure
    let .positive hε ← core q(inferInstance) q(inferInstance) ε | failure
    assertInstancesCommute
    pure (.positive q(triangleRemovalBound_pos $hε $hε₁))
  | _, _, _ => throwError "failed to match on Int.ceil application"


