/-- A `WeierstrassCurve` is in normal form of characteristic ≠ 2, if its $a_1, a_3 = 0$.
In other words it is $Y^2 = X^3 + a_2X^2 + a_4X + a_6$. -/
@[mk_iff]
class IsCharNeTwoNF : Prop where
  a₁ : W.a₁ = 0
  a₃ : W.a₃ = 0


@[simp]
theorem a₁_of_isCharNeTwoNF : W.a₁ = 0 := IsCharNeTwoNF.a₁


@[simp]
theorem a₃_of_isCharNeTwoNF : W.a₃ = 0 := IsCharNeTwoNF.a₃


@[simp]
theorem b₂_of_isCharNeTwoNF : W.b₂ = 4 * W.a₂ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.b₂ (HMul.hMul 4 W.a₂)
  -/
  rw [b₂, a₁_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HAdd.hAdd (HPow.hPow 0 2) (HMul.hMul 4 W.a₂)) (HMul.hMul 4 W.a₂)
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₄_of_isCharNeTwoNF : W.b₄ = 2 * W.a₄ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.b₄ (HMul.hMul 2 W.a₄)
  -/
  rw [b₄, a₃_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 W.a₄) (HMul.hMul W.a₁ 0)) (HMul.hMul 2 W.a₄)
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₆_of_isCharNeTwoNF : W.b₆ = 4 * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.b₆ (HMul.hMul 4 W.a₆)
  -/
  rw [b₆, a₃_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HAdd.hAdd (HPow.hPow 0 2) (HMul.hMul 4 W.a₆)) (HMul.hMul 4 W.a₆)
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₈_of_isCharNeTwoNF : W.b₈ = 4 * W.a₂ * W.a₆ - W.a₄ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.b₈ (HSub.hSub (HMul.hMul (HMul.hMul 4 W.a₂) W.a₆) (HPow.hPow W.a₄ 2))
  -/
  rw [b₈, a₁_of_isCharNeTwoNF, a₃_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow 0 2) W. …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem c₄_of_isCharNeTwoNF : W.c₄ = 16 * W.a₂ ^ 2 - 48 * W.a₄ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.c₄ (HSub.hSub (HMul.hMul 16 (HPow.hPow W.a₂ 2)) (HMul.hMul 48 W.a₄))
  -/
  rw [c₄, b₂_of_isCharNeTwoNF, b₄_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HSub.hSub (HPow.hPow (HMul.hMul 4 W.a₂) 2) (HMul.hMul 24 (HMul.hMul 2 W. …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem c₆_of_isCharNeTwoNF : W.c₆ = -64 * W.a₂ ^ 3 + 288 * W.a₂ * W.a₄ - 864 * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.c₆ (HSub.hSub (HAdd.hAdd (HMul.hMul (-64) (HPow.hPow W.a₂ 3)) (HMul.hMu …
  -/
  rw [c₆, b₂_of_isCharNeTwoNF, b₄_of_isCharNeTwoNF, b₆_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow (HMul.hMul 4 W.a₂) 3)) (HMul.hM …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem Δ_of_isCharNeTwoNF : W.Δ = -64 * W.a₂ ^ 3 * W.a₆ + 16 * W.a₂ ^ 2 * W.a₄ ^ 2 - 64 * W.a₄ ^ 3
    - 432 * W.a₆ ^ 2 + 288 * W.a₂ * W.a₄ * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq W.Δ (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul (-6 …
  -/
  rw [Δ, b₂_of_isCharNeTwoNF, b₄_of_isCharNeTwoNF, b₆_of_isCharNeTwoNF, b₈_of_isCharNeTwoNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharNeTwoNF
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow (HMul.hMu …
  -/
  ring1
  /-
    🎉 no goals
  -/


/-- There is an explicit change of variables of a `WeierstrassCurve` to
a normal form of characteristic ≠ 2, provided that 2 is invertible in the ring. -/
@[simps]
def toCharNeTwoNF : VariableChange R := ⟨1, 0, ⅟2 * -W.a₁, ⅟2 * -W.a₃⟩


instance toCharNeTwoNF_spec : (W.variableChange W.toCharNeTwoNF).IsCharNeTwoNF := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    F : Type u_2
    inst✝¹ : Field F
    W : WeierstrassCurve R
    inst✝ : Invertible 2
    ⊢ (W.variableChange W.toCharNeTwoNF).IsCharNeTwoNF
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp
                  /-
                    🎉 no goals
                  -/


theorem exists_variableChange_isCharNeTwoNF :
    ∃ C : VariableChange R, (W.variableChange C).IsCharNeTwoNF :=
  ⟨_, W.toCharNeTwoNF_spec⟩


/-- A `WeierstrassCurve` is in short normal form, if its $a_1, a_2, a_3 = 0$.
In other words it is $Y^2 = X^3 + a_4X + a_6$.

This is the normal form of characteristic ≠ 2 or 3, and
also the normal form of characteristic = 3 and j = 0. -/
@[mk_iff]
class IsShortNF : Prop where
  a₁ : W.a₁ = 0
  a₂ : W.a₂ = 0
  a₃ : W.a₃ = 0


instance isCharNeTwoNF_of_isShortNF : W.IsCharNeTwoNF := ⟨IsShortNF.a₁, IsShortNF.a₃⟩


theorem a₁_of_isShortNF : W.a₁ = 0 := IsShortNF.a₁


@[simp]
theorem a₂_of_isShortNF : W.a₂ = 0 := IsShortNF.a₂


theorem a₃_of_isShortNF : W.a₃ = 0 := IsShortNF.a₃


theorem b₂_of_isShortNF : W.b₂ = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsShortNF
    ⊢ Eq W.b₂ 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem b₄_of_isShortNF : W.b₄ = 2 * W.a₄ := W.b₄_of_isCharNeTwoNF


theorem b₆_of_isShortNF : W.b₆ = 4 * W.a₆ := W.b₆_of_isCharNeTwoNF


theorem b₈_of_isShortNF : W.b₈ = -W.a₄ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsShortNF
    ⊢ Eq W.b₈ (Neg.neg (HPow.hPow W.a₄ 2))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem c₄_of_isShortNF : W.c₄ = -48 * W.a₄ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsShortNF
    ⊢ Eq W.c₄ (HMul.hMul (-48) W.a₄)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem c₆_of_isShortNF : W.c₆ = -864 * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsShortNF
    ⊢ Eq W.c₆ (HMul.hMul (-864) W.a₆)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Δ_of_isShortNF : W.Δ = -16 * (4 * W.a₄ ^ 3 + 27 * W.a₆ ^ 2) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsShortNF
    ⊢ Eq W.Δ (HMul.hMul (-16) (HAdd.hAdd (HMul.hMul 4 (HPow.hPow W.a₄ 3)) (HMul.hM …
  -/
  rw [Δ_of_isCharNeTwoNF, a₂_of_isShortNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsShortNF
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul (-64) ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem b₄_of_isShortNF_of_char_three : W.b₄ = -W.a₄ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq W.b₄ (Neg.neg W.a₄)
  -/
  rw [b₄_of_isShortNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul 2 W.a₄) (Neg.neg W.a₄)
  -/
  linear_combination W.a₄ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem b₆_of_isShortNF_of_char_three : W.b₆ = W.a₆ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq W.b₆ W.a₆
  -/
  rw [b₆_of_isShortNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul 4 W.a₆) W.a₆
  -/
  linear_combination W.a₆ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem c₄_of_isShortNF_of_char_three : W.c₄ = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq W.c₄ 0
  -/
  rw [c₄_of_isShortNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul (-48) W.a₄) 0
  -/
  linear_combination -16 * W.a₄ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem c₆_of_isShortNF_of_char_three : W.c₆ = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq W.c₆ 0
  -/
  rw [c₆_of_isShortNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul (-864) W.a₆) 0
  -/
  linear_combination -288 * W.a₆ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem Δ_of_isShortNF_of_char_three : W.Δ = -W.a₄ ^ 3 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq W.Δ (Neg.neg (HPow.hPow W.a₄ 3))
  -/
  rw [Δ_of_isShortNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsShortNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul (-16) (HAdd.hAdd (HMul.hMul 4 (HPow.hPow W.a₄ 3)) (HMul.hMul 2 …
  -/
  linear_combination (-21 * W.a₄ ^ 3 - 144 * W.a₆ ^ 2) * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem j_of_isShortNF : W.j = 6912 * W.a₄ ^ 3 / (4 * W.a₄ ^ 3 + 27 * W.a₆ ^ 2) := by
  /-
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve F
    inst✝¹ : W.IsElliptic
    inst✝ : W.IsShortNF
    ⊢ Eq W.j (HDiv.hDiv (HMul.hMul 6912 (HPow.hPow W.a₄ 3)) (HAdd.hAdd (HMul.hMul  …
  -/
  have h := W.Δ'.ne_zero
  /-
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve F
    inst✝¹ : W.IsElliptic
    inst✝ : W.IsShortNF
    h : Ne (↑W.Δ') 0
    ⊢ Eq W.j (HDiv.hDiv (HMul.hMul 6912 (HPow.hPow W.a₄ 3)) (HAdd.hAdd (HMul.hMul  …
  -/
  rw [coe_Δ', Δ_of_isShortNF] at h
  rw [j, Units.val_inv_eq_inv_val, ← div_eq_inv_mul, coe_Δ',
    c₄_of_isShortNF, Δ_of_isShortNF, div_eq_div_iff h (right_ne_zero_of_mul h)]
  /-
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve F
    inst✝¹ : W.IsElliptic
    inst✝ : W.IsShortNF
    h : Ne (HMul.hMul (-16) (HAdd.hAdd (HMul.hMul 4 (HPow.hPow W.a₄ 3)) (HMul.hMul …
    ⊢ Eq (HMul.hMul (HPow.hPow (HMul.hMul (-48) W.a₄) 3) (HAdd.hAdd (HMul.hMul 4 ( …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem j_of_isShortNF_of_char_three [CharP F 3] : W.j = 0 := by
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsShortNF
    inst✝ : CharP F 3
    ⊢ Eq W.j 0
  -/
  rw [j, c₄_of_isShortNF_of_char_three]; simp
                                         /-
                                           🎉 no goals
                                         -/


/-- There is an explicit change of variables of a `WeierstrassCurve` to
a short normal form, provided that 2 and 3 are invertible in the ring.
It is the composition of an explicit change of variables with `WeierstrassCurve.toCharNeTwoNF`. -/
def toShortNF : VariableChange R :=
  .comp ⟨1, ⅟3 * -(W.variableChange W.toCharNeTwoNF).a₂, 0, 0⟩ W.toCharNeTwoNF


instance toShortNF_spec : (W.variableChange W.toShortNF).IsShortNF := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve R
    inst✝¹ : Invertible 2
    inst✝ : Invertible 3
    ⊢ (W.variableChange W.toShortNF).IsShortNF
  -/
  rw [toShortNF, variableChange_comp]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve R
    inst✝¹ : Invertible 2
    inst✝ : Invertible 3
    ⊢ ((W.variableChange W.toCharNeTwoNF).variableChange { u := 1, r := HMul.hMul  …
  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp
                  /-
                    🎉 no goals
                  -/


theorem exists_variableChange_isShortNF :
    ∃ C : VariableChange R, (W.variableChange C).IsShortNF :=
  ⟨_, W.toShortNF_spec⟩


/-- A `WeierstrassCurve` is in normal form of characteristic = 3 and j ≠ 0, if its
$a_1, a_3, a_4 = 0$. In other words it is $Y^2 = X^3 + a_2X^2 + a_6$. -/
@[mk_iff]
class IsCharThreeJNeZeroNF : Prop where
  a₁ : W.a₁ = 0
  a₃ : W.a₃ = 0
  a₄ : W.a₄ = 0


instance isCharNeTwoNF_of_isCharThreeJNeZeroNF : W.IsCharNeTwoNF :=
  ⟨IsCharThreeJNeZeroNF.a₁, IsCharThreeJNeZeroNF.a₃⟩


theorem a₁_of_isCharThreeJNeZeroNF : W.a₁ = 0 := IsCharThreeJNeZeroNF.a₁


theorem a₃_of_isCharThreeJNeZeroNF : W.a₃ = 0 := IsCharThreeJNeZeroNF.a₃


@[simp]
theorem a₄_of_isCharThreeJNeZeroNF : W.a₄ = 0 := IsCharThreeJNeZeroNF.a₄


theorem b₂_of_isCharThreeJNeZeroNF : W.b₂ = 4 * W.a₂ := W.b₂_of_isCharNeTwoNF


theorem b₄_of_isCharThreeJNeZeroNF : W.b₄ = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharThreeJNeZeroNF
    ⊢ Eq W.b₄ 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem b₆_of_isCharThreeJNeZeroNF : W.b₆ = 4 * W.a₆ := W.b₆_of_isCharNeTwoNF


theorem b₈_of_isCharThreeJNeZeroNF : W.b₈ = 4 * W.a₂ * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharThreeJNeZeroNF
    ⊢ Eq W.b₈ (HMul.hMul (HMul.hMul 4 W.a₂) W.a₆)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem c₄_of_isCharThreeJNeZeroNF : W.c₄ = 16 * W.a₂ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharThreeJNeZeroNF
    ⊢ Eq W.c₄ (HMul.hMul 16 (HPow.hPow W.a₂ 2))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem c₆_of_isCharThreeJNeZeroNF : W.c₆ = -64 * W.a₂ ^ 3 - 864 * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharThreeJNeZeroNF
    ⊢ Eq W.c₆ (HSub.hSub (HMul.hMul (-64) (HPow.hPow W.a₂ 3)) (HMul.hMul 864 W.a₆))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Δ_of_isCharThreeJNeZeroNF : W.Δ = -64 * W.a₂ ^ 3 * W.a₆ - 432 * W.a₆ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharThreeJNeZeroNF
    ⊢ Eq W.Δ (HSub.hSub (HMul.hMul (HMul.hMul (-64) (HPow.hPow W.a₂ 3)) W.a₆) (HMu …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem b₂_of_isCharThreeJNeZeroNF_of_char_three : W.b₂ = W.a₂ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq W.b₂ W.a₂
  -/
  rw [b₂_of_isCharThreeJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul 4 W.a₂) W.a₂
  -/
  linear_combination W.a₂ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem b₆_of_isCharThreeJNeZeroNF_of_char_three : W.b₆ = W.a₆ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq W.b₆ W.a₆
  -/
  rw [b₆_of_isCharThreeJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul 4 W.a₆) W.a₆
  -/
  linear_combination W.a₆ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem b₈_of_isCharThreeJNeZeroNF_of_char_three : W.b₈ = W.a₂ * W.a₆ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq W.b₈ (HMul.hMul W.a₂ W.a₆)
  -/
  rw [b₈_of_isCharThreeJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul (HMul.hMul 4 W.a₂) W.a₆) (HMul.hMul W.a₂ W.a₆)
  -/
  linear_combination W.a₂ * W.a₆ * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem c₄_of_isCharThreeJNeZeroNF_of_char_three : W.c₄ = W.a₂ ^ 2 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq W.c₄ (HPow.hPow W.a₂ 2)
  -/
  rw [c₄_of_isCharThreeJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul 16 (HPow.hPow W.a₂ 2)) (HPow.hPow W.a₂ 2)
  -/
  linear_combination 5 * W.a₂ ^ 2 * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem c₆_of_isCharThreeJNeZeroNF_of_char_three : W.c₆ = -W.a₂ ^ 3 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq W.c₆ (Neg.neg (HPow.hPow W.a₂ 3))
  -/
  rw [c₆_of_isCharThreeJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq (HSub.hSub (HMul.hMul (-64) (HPow.hPow W.a₂ 3)) (HMul.hMul 864 W.a₆)) (Ne …
  -/
  linear_combination (-21 * W.a₂ ^ 3 - 288 * W.a₆) * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem Δ_of_isCharThreeJNeZeroNF_of_char_three : W.Δ = -W.a₂ ^ 3 * W.a₆ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq W.Δ (HMul.hMul (Neg.neg (HPow.hPow W.a₂ 3)) W.a₆)
  -/
  rw [Δ_of_isCharThreeJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP R 3
    ⊢ Eq (HSub.hSub (HMul.hMul (HMul.hMul (-64) (HPow.hPow W.a₂ 3)) W.a₆) (HMul.hM …
  -/
  linear_combination (-21 * W.a₂ ^ 3 * W.a₆ - 144 * W.a₆ ^ 2) * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


@[simp]
theorem j_of_isCharThreeJNeZeroNF_of_char_three : W.j = -W.a₂ ^ 3 / W.a₆ := by
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP F 3
    ⊢ Eq W.j (HDiv.hDiv (Neg.neg (HPow.hPow W.a₂ 3)) W.a₆)
  -/
  have h := W.Δ'.ne_zero
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP F 3
    h : Ne (↑W.Δ') 0
    ⊢ Eq W.j (HDiv.hDiv (Neg.neg (HPow.hPow W.a₂ 3)) W.a₆)
  -/
  rw [coe_Δ', Δ_of_isCharThreeJNeZeroNF_of_char_three] at h
  rw [j, Units.val_inv_eq_inv_val, ← div_eq_inv_mul, coe_Δ',
    c₄_of_isCharThreeJNeZeroNF_of_char_three, Δ_of_isCharThreeJNeZeroNF_of_char_three,
    div_eq_div_iff h (right_ne_zero_of_mul h)]
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP F 3
    h : Ne (HMul.hMul (Neg.neg (HPow.hPow W.a₂ 3)) W.a₆) 0
    ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow W.a₂ 2) 3) W.a₆) (HMul.hMul (Neg.neg (HP …
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem j_ne_zero_of_isCharThreeJNeZeroNF_of_char_three : W.j ≠ 0 := by
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP F 3
    ⊢ Ne W.j 0
  -/
  rw [j_of_isCharThreeJNeZeroNF_of_char_three, div_ne_zero_iff]
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP F 3
    ⊢ And (Ne (Neg.neg (HPow.hPow W.a₂ 3)) 0) (Ne W.a₆ 0)
  -/
  have h := W.Δ'.ne_zero
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharThreeJNeZeroNF
    inst✝ : CharP F 3
    h : Ne (↑W.Δ') 0
    ⊢ And (Ne (Neg.neg (HPow.hPow W.a₂ 3)) 0) (Ne W.a₆ 0)
  -/
  rwa [coe_Δ', Δ_of_isCharThreeJNeZeroNF_of_char_three, mul_ne_zero_iff] at h
  /-
    🎉 no goals
  -/


/-- A `WeierstrassCurve` is in normal form of characteristic = 3, if it is
$Y^2 = X^3 + a_2X^2 + a_6$ (`WeierstrassCurve.IsCharThreeJNeZeroNF`) or
$Y^2 = X^3 + a_4X + a_6$ (`WeierstrassCurve.IsShortNF`). -/
class inductive IsCharThreeNF : Prop
| of_j_ne_zero [W.IsCharThreeJNeZeroNF] : IsCharThreeNF
| of_j_eq_zero [W.IsShortNF] : IsCharThreeNF


instance isCharThreeNF_of_isCharThreeJNeZeroNF [W.IsCharThreeJNeZeroNF] : W.IsCharThreeNF :=
  IsCharThreeNF.of_j_ne_zero


instance isCharThreeNF_of_isShortNF [W.IsShortNF] : W.IsCharThreeNF :=
  IsCharThreeNF.of_j_eq_zero


instance isCharNeTwoNF_of_isCharThreeNF [W.IsCharThreeNF] : W.IsCharNeTwoNF := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    F : Type u_2
    inst✝¹ : Field F
    W : WeierstrassCurve R
    inst✝ : W.IsCharThreeNF
    ⊢ W.IsCharNeTwoNF
  -/
                              /-
                                🎉 no goals
                              -/
  cases ‹W.IsCharThreeNF› <;> infer_instance
                              /-
                                🎉 no goals
                              -/


/-- For a `WeierstrassCurve` defined over a ring of characteristic = 3,
there is an explicit change of variables of it to $Y^2 = X^3 + a_4X + a_6$
(`WeierstrassCurve.IsShortNF`) if its j = 0.
This is in fact given by `WeierstrassCurve.toCharNeTwoNF`. -/
def toShortNFOfCharThree : VariableChange R :=
                                 /-
                                   R : Type u_1
                                   inst✝³ : CommRing R
                                   F : Type u_2
                                   inst✝² : Field F
                                   W : WeierstrassCurve R
                                   inst✝¹ : CharP R 3
                                   inst✝ : CharP F 3
                                   ⊢ Eq (HMul.hMul 2 2) 1
                                 -/
  have h : (2 : R) * 2 = 1 := by linear_combination CharP.cast_eq_zero R 3
                                 /-
                                   🎉 no goals
                                 -/
  letI : Invertible (2 : R) := ⟨2, h, h⟩
  W.toCharNeTwoNF


lemma toShortNFOfCharThree_a₂ : (W.variableChange W.toShortNFOfCharThree).a₂ = W.b₂ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (W.variableChange W.toShortNFOfCharThree).a₂ W.b₂
  -/
  simp_rw [toShortNFOfCharThree, toCharNeTwoNF, variableChange_a₂, inv_one, Units.val_one, b₂]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    ⊢ Eq (HMul.hMul (HPow.hPow 1 2) (HSub.hSub (HAdd.hAdd (HSub.hSub W.a₂ (HMul.hM …
  -/
  linear_combination (-W.a₂ - W.a₁ ^ 2) * CharP.cast_eq_zero R 3
  /-
    🎉 no goals
  -/


theorem toShortNFOfCharThree_spec (hb₂ : W.b₂ = 0) :
    (W.variableChange W.toShortNFOfCharThree).IsShortNF := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    hb₂ : Eq W.b₂ 0
    ⊢ (W.variableChange W.toShortNFOfCharThree).IsShortNF
  -/
  have h : (2 : R) * 2 = 1 := by linear_combination CharP.cast_eq_zero R 3
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    hb₂ : Eq W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    ⊢ (W.variableChange W.toShortNFOfCharThree).IsShortNF
  -/
  letI : Invertible (2 : R) := ⟨2, h, h⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    hb₂ : Eq W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    this : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
    ⊢ (W.variableChange W.toShortNFOfCharThree).IsShortNF
  -/
  have H := W.toCharNeTwoNF_spec
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 3
    hb₂ : Eq W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    this : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
    H : (W.variableChange W.toCharNeTwoNF).IsCharNeTwoNF
    ⊢ (W.variableChange W.toShortNFOfCharThree).IsShortNF
  -/
  exact ⟨H.a₁, hb₂ ▸ W.toShortNFOfCharThree_a₂, H.a₃⟩
  /-
    🎉 no goals
  -/


/-- For a `WeierstrassCurve` defined over a field of characteristic = 3,
there is an explicit change of variables of it to `WeierstrassCurve.IsCharThreeNF`, that is,
$Y^2 = X^3 + a_2X^2 + a_6$ (`WeierstrassCurve.IsCharThreeJNeZeroNF`) or
$Y^2 = X^3 + a_4X + a_6$ (`WeierstrassCurve.IsShortNF`).
It is the composition of an explicit change of variables with
`WeierstrassCurve.toShortNFOfCharThree`. -/
def toCharThreeNF : VariableChange F :=
  .comp ⟨1, (W.variableChange W.toShortNFOfCharThree).a₄ /
    (W.variableChange W.toShortNFOfCharThree).a₂, 0, 0⟩ W.toShortNFOfCharThree


theorem toCharThreeNF_spec_of_b₂_ne_zero (hb₂ : W.b₂ ≠ 0) :
    (W.variableChange W.toCharThreeNF).IsCharThreeJNeZeroNF := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Ne W.b₂ 0
    ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeJNeZeroNF
  -/
  have h : (2 : F) * 2 = 1 := by linear_combination CharP.cast_eq_zero F 3
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Ne W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeJNeZeroNF
  -/
  letI : Invertible (2 : F) := ⟨2, h, h⟩
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Ne W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    this : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
    ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeJNeZeroNF
  -/
  rw [toCharThreeNF, variableChange_comp]
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Ne W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    this : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
    ⊢ ((W.variableChange W.toShortNFOfCharThree).variableChange { u := 1, r := HDi …
  -/
  set W' := W.variableChange W.toShortNFOfCharThree
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Ne W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    this : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
    W' : WeierstrassCurve F := W.variableChange W.toShortNFOfCharThree
    ⊢ (W'.variableChange { u := 1, r := HDiv.hDiv W'.a₄ W'.a₂, s := 0, t := 0 }).I …
  -/
  haveI : W'.IsCharNeTwoNF := W.toCharNeTwoNF_spec
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Ne W.b₂ 0
    h : Eq (HMul.hMul 2 2) 1
    this✝ : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
    W' : WeierstrassCurve F := W.variableChange W.toShortNFOfCharThree
    this : W'.IsCharNeTwoNF
    ⊢ (W'.variableChange { u := 1, r := HDiv.hDiv W'.a₄ W'.a₂, s := 0, t := 0 }).I …
  -/
  constructor
    /-
      case a₁
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Ne W.b₂ 0
      h : Eq (HMul.hMul 2 2) 1
      this✝ : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
      W' : WeierstrassCurve F := W.variableChange W.toShortNFOfCharThree
      this : W'.IsCharNeTwoNF
      ⊢ Eq (W'.variableChange { u := 1, r := HDiv.hDiv W'.a₄ W'.a₂, s := 0, t := 0 } …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a₃
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Ne W.b₂ 0
      h : Eq (HMul.hMul 2 2) 1
      this✝ : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
      W' : WeierstrassCurve F := W.variableChange W.toShortNFOfCharThree
      this : W'.IsCharNeTwoNF
      ⊢ Eq (W'.variableChange { u := 1, r := HDiv.hDiv W'.a₄ W'.a₂, s := 0, t := 0 } …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a₄
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Ne W.b₂ 0
      h : Eq (HMul.hMul 2 2) 1
      this✝ : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
      W' : WeierstrassCurve F := W.variableChange W.toShortNFOfCharThree
      this : W'.IsCharNeTwoNF
      ⊢ Eq (W'.variableChange { u := 1, r := HDiv.hDiv W'.a₄ W'.a₂, s := 0, t := 0 } …
    -/
  · field_simp [W.toShortNFOfCharThree_a₂ ▸ hb₂]
    /-
      case a₄
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Ne W.b₂ 0
      h : Eq (HMul.hMul 2 2) 1
      this✝ : Invertible 2 := { invOf := 2, invOf_mul_self := h, mul_invOf_self := h }
      W' : WeierstrassCurve F := W.variableChange W.toShortNFOfCharThree
      this : W'.IsCharNeTwoNF
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd W'.a₄ (HMul.hMul 2 W'.a₄)) (HPow.hPow W' …
    -/
    linear_combination (W'.a₄ * W'.a₂ ^ 2 + W'.a₄ ^ 2) * CharP.cast_eq_zero F 3
    /-
      🎉 no goals
    -/


theorem toCharThreeNF_spec_of_b₂_eq_zero (hb₂ : W.b₂ = 0) :
    (W.variableChange W.toCharThreeNF).IsShortNF := by
  rw [toCharThreeNF, toShortNFOfCharThree_a₂, hb₂, div_zero, ← VariableChange.id,
    VariableChange.id_comp]
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    hb₂ : Eq W.b₂ 0
    ⊢ (W.variableChange W.toShortNFOfCharThree).IsShortNF
  -/
  exact W.toShortNFOfCharThree_spec hb₂
  /-
    🎉 no goals
  -/


instance toCharThreeNF_spec : (W.variableChange W.toCharThreeNF).IsCharThreeNF := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    F : Type u_2
    inst✝² : Field F
    W✝ : WeierstrassCurve R
    inst✝¹ : CharP R 3
    inst✝ : CharP F 3
    W : WeierstrassCurve F
    ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeNF
  -/
  by_cases hb₂ : W.b₂ = 0
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      F : Type u_2
      inst✝² : Field F
      W✝ : WeierstrassCurve R
      inst✝¹ : CharP R 3
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Eq W.b₂ 0
      ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeNF
    -/
  · haveI := W.toCharThreeNF_spec_of_b₂_eq_zero hb₂
    /-
      case pos
      R : Type u_1
      inst✝³ : CommRing R
      F : Type u_2
      inst✝² : Field F
      W✝ : WeierstrassCurve R
      inst✝¹ : CharP R 3
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Eq W.b₂ 0
      this : (W.variableChange W.toCharThreeNF).IsShortNF
      ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeNF
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      F : Type u_2
      inst✝² : Field F
      W✝ : WeierstrassCurve R
      inst✝¹ : CharP R 3
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Not (Eq W.b₂ 0)
      ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeNF
    -/
  · haveI := W.toCharThreeNF_spec_of_b₂_ne_zero hb₂
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      F : Type u_2
      inst✝² : Field F
      W✝ : WeierstrassCurve R
      inst✝¹ : CharP R 3
      inst✝ : CharP F 3
      W : WeierstrassCurve F
      hb₂ : Not (Eq W.b₂ 0)
      this : (W.variableChange W.toCharThreeNF).IsCharThreeJNeZeroNF
      ⊢ (W.variableChange W.toCharThreeNF).IsCharThreeNF
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem exists_variableChange_isCharThreeNF :
    ∃ C : VariableChange F, (W.variableChange C).IsCharThreeNF :=
  ⟨_, W.toCharThreeNF_spec⟩


/-- A `WeierstrassCurve` is in normal form of characteristic = 2 and j ≠ 0, if its $a_1 = 1$ and
$a_3, a_4 = 0$. In other words it is $Y^2 + XY = X^3 + a_2X^2 + a_6$. -/
@[mk_iff]
class IsCharTwoJNeZeroNF : Prop where
  a₁ : W.a₁ = 1
  a₃ : W.a₃ = 0
  a₄ : W.a₄ = 0


@[simp]
theorem a₁_of_isCharTwoJNeZeroNF : W.a₁ = 1 := IsCharTwoJNeZeroNF.a₁


@[simp]
theorem a₃_of_isCharTwoJNeZeroNF : W.a₃ = 0 := IsCharTwoJNeZeroNF.a₃


@[simp]
theorem a₄_of_isCharTwoJNeZeroNF : W.a₄ = 0 := IsCharTwoJNeZeroNF.a₄


@[simp]
theorem b₂_of_isCharTwoJNeZeroNF : W.b₂ = 1 + 4 * W.a₂ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq W.b₂ (HAdd.hAdd 1 (HMul.hMul 4 W.a₂))
  -/
  rw [b₂, a₁_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq (HAdd.hAdd (HPow.hPow 1 2) (HMul.hMul 4 W.a₂)) (HAdd.hAdd 1 (HMul.hMul 4  …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₄_of_isCharTwoJNeZeroNF : W.b₄ = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq W.b₄ 0
  -/
  rw [b₄, a₃_of_isCharTwoJNeZeroNF, a₄_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 0) (HMul.hMul W.a₁ 0)) 0
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₆_of_isCharTwoJNeZeroNF : W.b₆ = 4 * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq W.b₆ (HMul.hMul 4 W.a₆)
  -/
  rw [b₆, a₃_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq (HAdd.hAdd (HPow.hPow 0 2) (HMul.hMul 4 W.a₆)) (HMul.hMul 4 W.a₆)
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₈_of_isCharTwoJNeZeroNF : W.b₈ = W.a₆ + 4 * W.a₂ * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq W.b₈ (HAdd.hAdd W.a₆ (HMul.hMul (HMul.hMul 4 W.a₂) W.a₆))
  -/
  rw [b₈, a₁_of_isCharTwoJNeZeroNF, a₃_of_isCharTwoJNeZeroNF, a₄_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow 1 2) W. …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem c₄_of_isCharTwoJNeZeroNF : W.c₄ = W.b₂ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq W.c₄ (HPow.hPow W.b₂ 2)
  -/
  rw [c₄, b₄_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq (HSub.hSub (HPow.hPow W.b₂ 2) (HMul.hMul 24 0)) (HPow.hPow W.b₂ 2)
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem c₆_of_isCharTwoJNeZeroNF : W.c₆ = -W.b₂ ^ 3 - 864 * W.a₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq W.c₆ (HSub.hSub (Neg.neg (HPow.hPow W.b₂ 3)) (HMul.hMul 864 W.a₆))
  -/
  rw [c₆, b₄_of_isCharTwoJNeZeroNF, b₆_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJNeZeroNF
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow W.b₂ 3)) (HMul.hMul (HMul.hMul  …
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem b₂_of_isCharTwoJNeZeroNF_of_char_two : W.b₂ = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.b₂ 1
  -/
  rw [b₂_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd 1 (HMul.hMul 4 W.a₂)) 1
  -/
  linear_combination 2 * W.a₂ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem b₆_of_isCharTwoJNeZeroNF_of_char_two : W.b₆ = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.b₆ 0
  -/
  rw [b₆_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HMul.hMul 4 W.a₆) 0
  -/
  linear_combination 2 * W.a₆ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem b₈_of_isCharTwoJNeZeroNF_of_char_two : W.b₈ = W.a₆ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.b₈ W.a₆
  -/
  rw [b₈_of_isCharTwoJNeZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd W.a₆ (HMul.hMul (HMul.hMul 4 W.a₂) W.a₆)) W.a₆
  -/
  linear_combination 2 * W.a₂ * W.a₆ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem c₄_of_isCharTwoJNeZeroNF_of_char_two : W.c₄ = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.c₄ 1
  -/
  rw [c₄_of_isCharTwoJNeZeroNF, b₂_of_isCharTwoJNeZeroNF_of_char_two]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HPow.hPow 1 2) 1
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem c₆_of_isCharTwoJNeZeroNF_of_char_two : W.c₆ = 1 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.c₆ 1
  -/
  rw [c₆_of_isCharTwoJNeZeroNF, b₂_of_isCharTwoJNeZeroNF_of_char_two]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HSub.hSub (Neg.neg (HPow.hPow 1 3)) (HMul.hMul 864 W.a₆)) 1
  -/
  linear_combination (-1 - 432 * W.a₆) * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


@[simp]
theorem Δ_of_isCharTwoJNeZeroNF_of_char_two : W.Δ = W.a₆ := by
  rw [Δ, b₂_of_isCharTwoJNeZeroNF_of_char_two, b₄_of_isCharTwoJNeZeroNF,
    b₆_of_isCharTwoJNeZeroNF_of_char_two, b₈_of_isCharTwoJNeZeroNF_of_char_two]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow 1 2)) W.a …
  -/
  linear_combination -W.a₆ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


@[simp]
theorem j_of_isCharTwoJNeZeroNF_of_char_two : W.j = 1 / W.a₆ := by
  rw [j, Units.val_inv_eq_inv_val, ← div_eq_inv_mul, coe_Δ',
    c₄_of_isCharTwoJNeZeroNF_of_char_two, Δ_of_isCharTwoJNeZeroNF_of_char_two, one_pow]


theorem j_ne_zero_of_isCharTwoJNeZeroNF_of_char_two : W.j ≠ 0 := by
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP F 2
    ⊢ Ne W.j 0
  -/
  rw [j_of_isCharTwoJNeZeroNF_of_char_two, div_ne_zero_iff]
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP F 2
    ⊢ And (Ne 1 0) (Ne W.a₆ 0)
  -/
  have h := W.Δ'.ne_zero
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP F 2
    h : Ne (↑W.Δ') 0
    ⊢ And (Ne 1 0) (Ne W.a₆ 0)
  -/
  rw [coe_Δ', Δ_of_isCharTwoJNeZeroNF_of_char_two] at h
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharTwoJNeZeroNF
    inst✝ : CharP F 2
    h : Ne W.a₆ 0
    ⊢ And (Ne 1 0) (Ne W.a₆ 0)
  -/
  exact ⟨one_ne_zero, h⟩
  /-
    🎉 no goals
  -/


/-- A `WeierstrassCurve` is in normal form of characteristic = 2 and j = 0, if its $a_1, a_2 = 0$.
In other words it is $Y^2 + a_3Y = X^3 + a_4X + a_6$. -/
@[mk_iff]
class IsCharTwoJEqZeroNF : Prop where
  a₁ : W.a₁ = 0
  a₂ : W.a₂ = 0


@[simp]
theorem a₁_of_isCharTwoJEqZeroNF : W.a₁ = 0 := IsCharTwoJEqZeroNF.a₁


@[simp]
theorem a₂_of_isCharTwoJEqZeroNF : W.a₂ = 0 := IsCharTwoJEqZeroNF.a₂


@[simp]
theorem b₂_of_isCharTwoJEqZeroNF : W.b₂ = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.b₂ 0
  -/
  rw [b₂, a₁_of_isCharTwoJEqZeroNF, a₂_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq (HAdd.hAdd (HPow.hPow 0 2) (HMul.hMul 4 0)) 0
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₄_of_isCharTwoJEqZeroNF : W.b₄ = 2 * W.a₄ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.b₄ (HMul.hMul 2 W.a₄)
  -/
  rw [b₄, a₁_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 W.a₄) (HMul.hMul 0 W.a₃)) (HMul.hMul 2 W.a₄)
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem b₈_of_isCharTwoJEqZeroNF : W.b₈ = -W.a₄ ^ 2 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.b₈ (Neg.neg (HPow.hPow W.a₄ 2))
  -/
  rw [b₈, a₁_of_isCharTwoJEqZeroNF, a₂_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HMul.hMul (HPow.hPow 0 2) W. …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem c₄_of_isCharTwoJEqZeroNF : W.c₄ = -48 * W.a₄ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.c₄ (HMul.hMul (-48) W.a₄)
  -/
  rw [c₄, b₂_of_isCharTwoJEqZeroNF, b₄_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq (HSub.hSub (HPow.hPow 0 2) (HMul.hMul 24 (HMul.hMul 2 W.a₄))) (HMul.hMul  …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem c₆_of_isCharTwoJEqZeroNF : W.c₆ = -216 * W.b₆ := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.c₆ (HMul.hMul (-216) W.b₆)
  -/
  rw [c₆, b₂_of_isCharTwoJEqZeroNF, b₄_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Neg.neg (HPow.hPow 0 3)) (HMul.hMul (HMul.hMul 36  …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem Δ_of_isCharTwoJEqZeroNF : W.Δ = -(64 * W.a₄ ^ 3 + 27 * W.b₆ ^ 2) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.Δ (Neg.neg (HAdd.hAdd (HMul.hMul 64 (HPow.hPow W.a₄ 3)) (HMul.hMul 27 ( …
  -/
  rw [Δ, b₂_of_isCharTwoJEqZeroNF, b₄_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Neg.neg (HPow.hPow 0 2)) W.b …
  -/
  ring1
  /-
    🎉 no goals
  -/


theorem b₄_of_isCharTwoJEqZeroNF_of_char_two : W.b₄ = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.b₄ 0
  -/
  rw [b₄_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HMul.hMul 2 W.a₄) 0
  -/
  linear_combination W.a₄ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem b₈_of_isCharTwoJEqZeroNF_of_char_two : W.b₈ = W.a₄ ^ 2 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.b₈ (HPow.hPow W.a₄ 2)
  -/
  rw [b₈_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (Neg.neg (HPow.hPow W.a₄ 2)) (HPow.hPow W.a₄ 2)
  -/
  linear_combination -W.a₄ ^ 2 * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem c₄_of_isCharTwoJEqZeroNF_of_char_two : W.c₄ = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.c₄ 0
  -/
  rw [c₄_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HMul.hMul (-48) W.a₄) 0
  -/
  linear_combination -24 * W.a₄ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem c₆_of_isCharTwoJEqZeroNF_of_char_two : W.c₆ = 0 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.c₆ 0
  -/
  rw [c₆_of_isCharTwoJEqZeroNF]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (HMul.hMul (-216) W.b₆) 0
  -/
  linear_combination -108 * W.b₆ * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem Δ_of_isCharTwoJEqZeroNF_of_char_two : W.Δ = W.a₃ ^ 4 := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq W.Δ (HPow.hPow W.a₃ 4)
  -/
  rw [Δ_of_isCharTwoJEqZeroNF, b₆_of_char_two]
  /-
    R : Type u_1
    inst✝² : CommRing R
    W : WeierstrassCurve R
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP R 2
    ⊢ Eq (Neg.neg (HAdd.hAdd (HMul.hMul 64 (HPow.hPow W.a₄ 3)) (HMul.hMul 27 (HPow …
  -/
  linear_combination (-32 * W.a₄ ^ 3 - 14 * W.a₃ ^ 4) * CharP.cast_eq_zero R 2
  /-
    🎉 no goals
  -/


theorem j_of_isCharTwoJEqZeroNF : W.j = 110592 * W.a₄ ^ 3 / (64 * W.a₄ ^ 3 + 27 * W.b₆ ^ 2) := by
  /-
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve F
    inst✝¹ : W.IsElliptic
    inst✝ : W.IsCharTwoJEqZeroNF
    ⊢ Eq W.j (HDiv.hDiv (HMul.hMul 110592 (HPow.hPow W.a₄ 3)) (HAdd.hAdd (HMul.hMu …
  -/
  have h := W.Δ'.ne_zero
  /-
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve F
    inst✝¹ : W.IsElliptic
    inst✝ : W.IsCharTwoJEqZeroNF
    h : Ne (↑W.Δ') 0
    ⊢ Eq W.j (HDiv.hDiv (HMul.hMul 110592 (HPow.hPow W.a₄ 3)) (HAdd.hAdd (HMul.hMu …
  -/
  rw [coe_Δ', Δ_of_isCharTwoJEqZeroNF] at h
  rw [j, Units.val_inv_eq_inv_val, ← div_eq_inv_mul, coe_Δ',
    c₄_of_isCharTwoJEqZeroNF, Δ_of_isCharTwoJEqZeroNF, div_eq_div_iff h (neg_ne_zero.1 h)]
  /-
    F : Type u_2
    inst✝² : Field F
    W : WeierstrassCurve F
    inst✝¹ : W.IsElliptic
    inst✝ : W.IsCharTwoJEqZeroNF
    h : Ne (Neg.neg (HAdd.hAdd (HMul.hMul 64 (HPow.hPow W.a₄ 3)) (HMul.hMul 27 (HP …
    ⊢ Eq (HMul.hMul (HPow.hPow (HMul.hMul (-48) W.a₄) 3) (HAdd.hAdd (HMul.hMul 64  …
  -/
  ring1
  /-
    🎉 no goals
  -/


@[simp]
theorem j_of_isCharTwoJEqZeroNF_of_char_two [CharP F 2] : W.j = 0 := by
  /-
    F : Type u_2
    inst✝³ : Field F
    W : WeierstrassCurve F
    inst✝² : W.IsElliptic
    inst✝¹ : W.IsCharTwoJEqZeroNF
    inst✝ : CharP F 2
    ⊢ Eq W.j 0
  -/
  rw [j, c₄_of_isCharTwoJEqZeroNF_of_char_two]; simp
                                                /-
                                                  🎉 no goals
                                                -/


/-- A `WeierstrassCurve` is in normal form of characteristic = 2, if it is
$Y^2 + XY = X^3 + a_2X^2 + a_6$ (`WeierstrassCurve.IsCharTwoJNeZeroNF`) or
$Y^2 + a_3Y = X^3 + a_4X + a_6$ (`WeierstrassCurve.IsCharTwoJEqZeroNF`). -/
class inductive IsCharTwoNF : Prop
| of_j_ne_zero [W.IsCharTwoJNeZeroNF] : IsCharTwoNF
| of_j_eq_zero [W.IsCharTwoJEqZeroNF] : IsCharTwoNF


instance isCharTwoNF_of_isCharTwoJNeZeroNF [W.IsCharTwoJNeZeroNF] : W.IsCharTwoNF :=
  IsCharTwoNF.of_j_ne_zero


instance isCharTwoNF_of_isCharTwoJEqZeroNF [W.IsCharTwoJEqZeroNF] : W.IsCharTwoNF :=
  IsCharTwoNF.of_j_eq_zero


/-- For a `WeierstrassCurve` defined over a ring of characteristic = 2,
there is an explicit change of variables of it to $Y^2 + a_3Y = X^3 + a_4X + a_6$
(`WeierstrassCurve.IsCharTwoJEqZeroNF`) if its j = 0. -/
def toCharTwoJEqZeroNF : VariableChange R := ⟨1, W.a₂, 0, 0⟩


theorem toCharTwoJEqZeroNF_spec (ha₁ : W.a₁ = 0) :
    (W.variableChange W.toCharTwoJEqZeroNF).IsCharTwoJEqZeroNF := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    W : WeierstrassCurve R
    inst✝ : CharP R 2
    ha₁ : Eq W.a₁ 0
    ⊢ (W.variableChange W.toCharTwoJEqZeroNF).IsCharTwoJEqZeroNF
  -/
  constructor
    /-
      case a₁
      R : Type u_1
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 2
      ha₁ : Eq W.a₁ 0
      ⊢ Eq (W.variableChange W.toCharTwoJEqZeroNF).a₁ 0
    -/
  · simp [toCharTwoJEqZeroNF, ha₁]
    /-
      🎉 no goals
    -/
    /-
      case a₂
      R : Type u_1
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 2
      ha₁ : Eq W.a₁ 0
      ⊢ Eq (W.variableChange W.toCharTwoJEqZeroNF).a₂ 0
    -/
  · simp_rw [toCharTwoJEqZeroNF, variableChange_a₂, inv_one, Units.val_one]
    /-
      case a₂
      R : Type u_1
      inst✝¹ : CommRing R
      W : WeierstrassCurve R
      inst✝ : CharP R 2
      ha₁ : Eq W.a₁ 0
      ⊢ Eq (HMul.hMul (HPow.hPow 1 2) (HSub.hSub (HAdd.hAdd (HSub.hSub W.a₂ (HMul.hM …
    -/
    linear_combination 2 * W.a₂ * CharP.cast_eq_zero R 2
    /-
      🎉 no goals
    -/


/-- For a `WeierstrassCurve` defined over a field of characteristic = 2,
there is an explicit change of variables of it to $Y^2 + XY = X^3 + a_2X^2 + a_6$
(`WeierstrassCurve.IsCharTwoJNeZeroNF`) if its j ≠ 0. -/
def toCharTwoJNeZeroNF (W : WeierstrassCurve F) (ha₁ : W.a₁ ≠ 0) : VariableChange F :=
  ⟨Units.mk0 _ ha₁, W.a₃ / W.a₁, 0, (W.a₁ ^ 2 * W.a₄ + W.a₃ ^ 2) / W.a₁ ^ 3⟩


theorem toCharTwoJNeZeroNF_spec (ha₁ : W.a₁ ≠ 0) :
    (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).IsCharTwoJNeZeroNF := by
  /-
    F : Type u_2
    inst✝¹ : Field F
    inst✝ : CharP F 2
    W : WeierstrassCurve F
    ha₁ : Ne W.a₁ 0
    ⊢ (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).IsCharTwoJNeZeroNF
  -/
  constructor
    /-
      case a₁
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 2
      W : WeierstrassCurve F
      ha₁ : Ne W.a₁ 0
      ⊢ Eq (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).a₁ 1
    -/
  · simp [toCharTwoJNeZeroNF, ha₁]
    /-
      🎉 no goals
    -/
    /-
      case a₃
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 2
      W : WeierstrassCurve F
      ha₁ : Ne W.a₁ 0
      ⊢ Eq (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).a₃ 0
    -/
  · field_simp [toCharTwoJNeZeroNF]
    /-
      case a₃
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 2
      W : WeierstrassCurve F
      ha₁ : Ne W.a₁ 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd W.a₃ W.a₃) (HPow.hPow W.a₁ 3)) (HMul.hMu …
    -/
    linear_combination (W.a₃ * W.a₁ ^ 3 + W.a₁ ^ 2 * W.a₄ + W.a₃ ^ 2) * CharP.cast_eq_zero F 2
    /-
      🎉 no goals
    -/
    /-
      case a₄
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 2
      W : WeierstrassCurve F
      ha₁ : Ne W.a₁ 0
      ⊢ Eq (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).a₄ 0
    -/
  · field_simp [toCharTwoJNeZeroNF]
    /-
      case a₄
      F : Type u_2
      inst✝¹ : Field F
      inst✝ : CharP F 2
      W : WeierstrassCurve F
      ha₁ : Ne W.a₁ 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (HAdd.hAdd (HMul.hMul W.a₄ W. …
    -/
    linear_combination (W.a₁ ^ 4 * W.a₃ ^ 2 + W.a₁ ^ 5 * W.a₃ * W.a₂) * CharP.cast_eq_zero F 2
    /-
      🎉 no goals
    -/


/-- For a `WeierstrassCurve` defined over a field of characteristic = 2,
there is an explicit change of variables of it to `WeierstrassCurve.IsCharTwoNF`, that is,
$Y^2 + XY = X^3 + a_2X^2 + a_6$ (`WeierstrassCurve.IsCharTwoJNeZeroNF`) or
$Y^2 + a_3Y = X^3 + a_4X + a_6$ (`WeierstrassCurve.IsCharTwoJEqZeroNF`). -/
def toCharTwoNF [DecidableEq F] : VariableChange F :=
  if ha₁ : W.a₁ = 0 then W.toCharTwoJEqZeroNF else W.toCharTwoJNeZeroNF ha₁


instance toCharTwoNF_spec [DecidableEq F] : (W.variableChange W.toCharTwoNF).IsCharTwoNF := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    F : Type u_2
    inst✝³ : Field F
    W✝ : WeierstrassCurve R
    inst✝² : CharP R 2
    inst✝¹ : CharP F 2
    W : WeierstrassCurve F
    inst✝ : DecidableEq F
    ⊢ (W.variableChange W.toCharTwoNF).IsCharTwoNF
  -/
  by_cases ha₁ : W.a₁ = 0
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      F : Type u_2
      inst✝³ : Field F
      W✝ : WeierstrassCurve R
      inst✝² : CharP R 2
      inst✝¹ : CharP F 2
      W : WeierstrassCurve F
      inst✝ : DecidableEq F
      ha₁ : Eq W.a₁ 0
      ⊢ (W.variableChange W.toCharTwoNF).IsCharTwoNF
    -/
  · rw [toCharTwoNF, dif_pos ha₁]
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      F : Type u_2
      inst✝³ : Field F
      W✝ : WeierstrassCurve R
      inst✝² : CharP R 2
      inst✝¹ : CharP F 2
      W : WeierstrassCurve F
      inst✝ : DecidableEq F
      ha₁ : Eq W.a₁ 0
      ⊢ (W.variableChange W.toCharTwoJEqZeroNF).IsCharTwoNF
    -/
    haveI := W.toCharTwoJEqZeroNF_spec ha₁
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      F : Type u_2
      inst✝³ : Field F
      W✝ : WeierstrassCurve R
      inst✝² : CharP R 2
      inst✝¹ : CharP F 2
      W : WeierstrassCurve F
      inst✝ : DecidableEq F
      ha₁ : Eq W.a₁ 0
      this : (W.variableChange W.toCharTwoJEqZeroNF).IsCharTwoJEqZeroNF
      ⊢ (W.variableChange W.toCharTwoJEqZeroNF).IsCharTwoNF
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      F : Type u_2
      inst✝³ : Field F
      W✝ : WeierstrassCurve R
      inst✝² : CharP R 2
      inst✝¹ : CharP F 2
      W : WeierstrassCurve F
      inst✝ : DecidableEq F
      ha₁ : Not (Eq W.a₁ 0)
      ⊢ (W.variableChange W.toCharTwoNF).IsCharTwoNF
    -/
  · rw [toCharTwoNF, dif_neg ha₁]
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      F : Type u_2
      inst✝³ : Field F
      W✝ : WeierstrassCurve R
      inst✝² : CharP R 2
      inst✝¹ : CharP F 2
      W : WeierstrassCurve F
      inst✝ : DecidableEq F
      ha₁ : Not (Eq W.a₁ 0)
      ⊢ (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).IsCharTwoNF
    -/
    haveI := W.toCharTwoJNeZeroNF_spec ha₁
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      F : Type u_2
      inst✝³ : Field F
      W✝ : WeierstrassCurve R
      inst✝² : CharP R 2
      inst✝¹ : CharP F 2
      W : WeierstrassCurve F
      inst✝ : DecidableEq F
      ha₁ : Not (Eq W.a₁ 0)
      this : (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).IsCharTwoJNeZeroNF
      ⊢ (W.variableChange (W.toCharTwoJNeZeroNF ha₁)).IsCharTwoNF
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem exists_variableChange_isCharTwoNF :
    ∃ C : VariableChange F, (W.variableChange C).IsCharTwoNF := by
  classical
  exact ⟨_, W.toCharTwoNF_spec⟩


