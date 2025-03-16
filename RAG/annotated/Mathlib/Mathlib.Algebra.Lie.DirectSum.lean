instance : LieRingModule L (⨁ i, M i) where
  bracket x m := m.mapRange (fun _ m' => ⁅x, m'⁆) fun _ => lie_zero x
  add_lie x y m := by
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      x y : L
      m : DirectSum ι fun i => M i
      ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) m) (HAdd.hAdd (Bracket.bracket x m) (Bra …
    -/
    refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      x y : L
      m : DirectSum ι fun i => M i
      x✝ : ι
      ⊢ Eq ((Bracket.bracket (HAdd.hAdd x y) m) x✝) ((HAdd.hAdd (Bracket.bracket x m …
    -/
    simp only [mapRange_apply, add_apply, add_lie]
    /-
      🎉 no goals
    -/
  lie_add x m n := by
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      x : L
      m n : DirectSum ι fun i => M i
      ⊢ Eq (Bracket.bracket x (HAdd.hAdd m n)) (HAdd.hAdd (Bracket.bracket x m) (Bra …
    -/
    refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      x : L
      m n : DirectSum ι fun i => M i
      x✝ : ι
      ⊢ Eq ((Bracket.bracket x (HAdd.hAdd m n)) x✝) ((HAdd.hAdd (Bracket.bracket x m …
    -/
    simp only [mapRange_apply, add_apply, lie_add]
    /-
      🎉 no goals
    -/
  leibniz_lie x y m := by
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      x y : L
      m : DirectSum ι fun i => M i
      ⊢ Eq (Bracket.bracket x (Bracket.bracket y m)) (HAdd.hAdd (Bracket.bracket (Br …
    -/
    refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      x y : L
      m : DirectSum ι fun i => M i
      x✝ : ι
      ⊢ Eq ((Bracket.bracket x (Bracket.bracket y m)) x✝) ((HAdd.hAdd (Bracket.brack …
    -/
    simp only [mapRange_apply, lie_lie, add_apply, sub_add_cancel]
    /-
      🎉 no goals
    -/


@[simp]
theorem lie_module_bracket_apply (x : L) (m : ⨁ i, M i) (i : ι) : ⁅x, m⁆ i = ⁅x, m i⁆ :=
  mapRange_apply _ _ m i


instance : LieModule R L (⨁ i, M i) where
  smul_lie t x m := by
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      t : R
      x : L
      m : DirectSum ι fun i => M i
      ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext i`
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      t : R
      x : L
      m : DirectSum ι fun i => M i
      x✝ : ι
      ⊢ Eq ((Bracket.bracket (HSMul.hSMul t x) m) x✝) ((HSMul.hSMul t (Bracket.brack …
    -/
    simp only [smul_lie, lie_module_bracket_apply, smul_apply]
    /-
      🎉 no goals
    -/
  lie_smul t x m := by
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      t : R
      x : L
      m : DirectSum ι fun i => M i
      ⊢ Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext i`
    /-
      R : Type u
      ι : Type v
      inst✝⁶ : CommRing R
      L : Type w₁
      M : ι → Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : (i : ι) → AddCommGroup (M i)
      inst✝² : (i : ι) → Module R (M i)
      inst✝¹ : (i : ι) → LieRingModule L (M i)
      inst✝ : ∀ (i : ι), LieModule R L (M i)
      t : R
      x : L
      m : DirectSum ι fun i => M i
      x✝ : ι
      ⊢ Eq ((Bracket.bracket x (HSMul.hSMul t m)) x✝) ((HSMul.hSMul t (Bracket.brack …
    -/
    simp only [lie_smul, lie_module_bracket_apply, smul_apply]
    /-
      🎉 no goals
    -/


/-- The inclusion of each component into a direct sum as a morphism of Lie modules. -/
def lieModuleOf [DecidableEq ι] (j : ι) : M j →ₗ⁅R,L⁆ ⨁ i, M i :=
  { lof R ι M j with
    map_lie' := fun {x m} => by
      /-
        R : Type u
        ι : Type v
        inst✝⁷ : CommRing R
        L : Type w₁
        M : ι → Type w
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : (i : ι) → AddCommGroup (M i)
        inst✝³ : (i : ι) → Module R (M i)
        inst✝² : (i : ι) → LieRingModule L (M i)
        inst✝¹ : ∀ (i : ι), LieModule R L (M i)
        inst✝ : DecidableEq ι
        j : ι
        x : L
        m : M j
        ⊢ Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket x (__src✝.toFun m))
      -/
      refine DFinsupp.ext fun i => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext i`
      /-
        R : Type u
        ι : Type v
        inst✝⁷ : CommRing R
        L : Type w₁
        M : ι → Type w
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra R L
        inst✝⁴ : (i : ι) → AddCommGroup (M i)
        inst✝³ : (i : ι) → Module R (M i)
        inst✝² : (i : ι) → LieRingModule L (M i)
        inst✝¹ : ∀ (i : ι), LieModule R L (M i)
        inst✝ : DecidableEq ι
        j : ι
        x : L
        m : M j
        i : ι
        ⊢ Eq ((__src✝.toFun (Bracket.bracket x m)) i) ((Bracket.bracket x (__src✝.toFu …
      -/
      by_cases h : j = i
        /-
          case pos
          R : Type u
          ι : Type v
          inst✝⁷ : CommRing R
          L : Type w₁
          M : ι → Type w
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          inst✝⁴ : (i : ι) → AddCommGroup (M i)
          inst✝³ : (i : ι) → Module R (M i)
          inst✝² : (i : ι) → LieRingModule L (M i)
          inst✝¹ : ∀ (i : ι), LieModule R L (M i)
          inst✝ : DecidableEq ι
          j : ι
          x : L
          m : M j
          i : ι
          h : Eq j i
          ⊢ Eq ((__src✝.toFun (Bracket.bracket x m)) i) ((Bracket.bracket x (__src✝.toFu …
        -/
      · rw [← h]; simp
                  /-
                    🎉 no goals
                  -/
      · -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
        -- old proof `simp [lof, lsingle, h]`
        /-
          case neg
          R : Type u
          ι : Type v
          inst✝⁷ : CommRing R
          L : Type w₁
          M : ι → Type w
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          inst✝⁴ : (i : ι) → AddCommGroup (M i)
          inst✝³ : (i : ι) → Module R (M i)
          inst✝² : (i : ι) → LieRingModule L (M i)
          inst✝¹ : ∀ (i : ι), LieModule R L (M i)
          inst✝ : DecidableEq ι
          j : ι
          x : L
          m : M j
          i : ι
          h : Not (Eq j i)
          ⊢ Eq ((__src✝.toFun (Bracket.bracket x m)) i) ((Bracket.bracket x (__src✝.toFu …
        -/
        simp only [lof, lsingle, AddHom.toFun_eq_coe, lie_module_bracket_apply]
        /-
          case neg
          R : Type u
          ι : Type v
          inst✝⁷ : CommRing R
          L : Type w₁
          M : ι → Type w
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          inst✝⁴ : (i : ι) → AddCommGroup (M i)
          inst✝³ : (i : ι) → Module R (M i)
          inst✝² : (i : ι) → LieRingModule L (M i)
          inst✝¹ : ∀ (i : ι), LieModule R L (M i)
          inst✝ : DecidableEq ι
          j : ι
          x : L
          m : M j
          i : ι
          h : Not (Eq j i)
          ⊢ Eq (({ toFun := DFinsupp.single j, map_add' := ⋯ } (Bracket.bracket x m)) i) …
        -/
        erw [AddHom.coe_mk]
        /-
          case neg
          R : Type u
          ι : Type v
          inst✝⁷ : CommRing R
          L : Type w₁
          M : ι → Type w
          inst✝⁶ : LieRing L
          inst✝⁵ : LieAlgebra R L
          inst✝⁴ : (i : ι) → AddCommGroup (M i)
          inst✝³ : (i : ι) → Module R (M i)
          inst✝² : (i : ι) → LieRingModule L (M i)
          inst✝¹ : ∀ (i : ι), LieModule R L (M i)
          inst✝ : DecidableEq ι
          j : ι
          x : L
          m : M j
          i : ι
          h : Not (Eq j i)
          ⊢ Eq ((DFinsupp.single j (Bracket.bracket x m)) i) (Bracket.bracket x ((DFinsu …
        -/
        simp [h] }
        /-
          🎉 no goals
        -/


/-- The projection map onto one component, as a morphism of Lie modules. -/
def lieModuleComponent (j : ι) : (⨁ i, M i) →ₗ⁅R,L⁆ M j :=
  { component R ι M j with
                                /-
                                  R : Type u
                                  ι : Type v
                                  inst✝⁶ : CommRing R
                                  L : Type w₁
                                  M : ι → Type w
                                  inst✝⁵ : LieRing L
                                  inst✝⁴ : LieAlgebra R L
                                  inst✝³ : (i : ι) → AddCommGroup (M i)
                                  inst✝² : (i : ι) → Module R (M i)
                                  inst✝¹ : (i : ι) → LieRingModule L (M i)
                                  inst✝ : ∀ (i : ι), LieModule R L (M i)
                                  j : ι
                                  x : L
                                  m : DirectSum ι fun i => M i
                                  ⊢ Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket x (__src✝.toFun m))
                                -/
    map_lie' := fun {x m} => by simp [component, lapply] }
                                /-
                                  🎉 no goals
                                -/


instance lieRing : LieRing (⨁ i, L i) :=
  { (inferInstance : AddCommGroup _) with
    bracket := zipWith (fun _ => fun x y => ⁅x, y⁆) fun _ => lie_zero 0
    add_lie := fun x y z => by
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) z) (HAdd.hAdd (Bracket.bracket x z) (Bra …
      -/
      refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        x✝ : ι
        ⊢ Eq ((Bracket.bracket (HAdd.hAdd x y) z) x✝) ((HAdd.hAdd (Bracket.bracket x z …
      -/
      simp only [zipWith_apply, add_apply, add_lie]
      /-
        🎉 no goals
      -/
    lie_add := fun x y z => by
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        ⊢ Eq (Bracket.bracket x (HAdd.hAdd y z)) (HAdd.hAdd (Bracket.bracket x y) (Bra …
      -/
      refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        x✝ : ι
        ⊢ Eq ((Bracket.bracket x (HAdd.hAdd y z)) x✝) ((HAdd.hAdd (Bracket.bracket x y …
      -/
      simp only [zipWith_apply, add_apply, lie_add]
      /-
        🎉 no goals
      -/
    lie_self := fun x => by
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x : DirectSum ι fun i => L i
        ⊢ Eq (Bracket.bracket x x) 0
      -/
      refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x : DirectSum ι fun i => L i
        x✝ : ι
        ⊢ Eq ((Bracket.bracket x x) x✝) (0 x✝)
      -/
      simp only [zipWith_apply, add_apply, lie_self, zero_apply]
      /-
        🎉 no goals
      -/
    leibniz_lie := fun x y z => by
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        ⊢ Eq (Bracket.bracket x (Bracket.bracket y z)) (HAdd.hAdd (Bracket.bracket (Br …
      -/
      refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        x✝ : ι
        ⊢ Eq ((Bracket.bracket x (Bracket.bracket y z)) x✝) ((HAdd.hAdd (Bracket.brack …
      -/
      simp only [sub_apply, zipWith_apply, add_apply, zero_apply]
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        x y z : DirectSum ι fun i => L i
        x✝ : ι
        ⊢ Eq (Bracket.bracket (x x✝) (Bracket.bracket (y x✝) (z x✝))) (HAdd.hAdd (Brac …
      -/
      apply leibniz_lie }
      /-
        🎉 no goals
      -/


@[simp]
theorem bracket_apply (x y : ⨁ i, L i) (i : ι) : ⁅x, y⁆ i = ⁅x i, y i⁆ :=
  zipWith_apply _ _ x y i


theorem lie_of_same [DecidableEq ι] {i : ι} (x y : L i) :
    ⁅of L i x, of L i y⁆ = of L i ⁅x, y⁆ :=
  DFinsupp.zipWith_single_single _ _ _ _


theorem lie_of_of_ne [DecidableEq ι] {i j : ι} (hij : i ≠ j) (x : L i) (y : L j) :
    ⁅of L i x, of L j y⁆ = 0 := by
  /-
    ι : Type v
    L : ι → Type w
    inst✝¹ : (i : ι) → LieRing (L i)
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    x : L i
    y : L j
    ⊢ Eq (Bracket.bracket ((DirectSum.of L i) x) ((DirectSum.of L j) y)) 0
  -/
  refine DFinsupp.ext fun k => ?_
  /-
    ι : Type v
    L : ι → Type w
    inst✝¹ : (i : ι) → LieRing (L i)
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    x : L i
    y : L j
    k : ι
    ⊢ Eq ((Bracket.bracket ((DirectSum.of L i) x) ((DirectSum.of L j) y)) k) (0 k)
  -/
  rw [bracket_apply]
  /-
    ι : Type v
    L : ι → Type w
    inst✝¹ : (i : ι) → LieRing (L i)
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    x : L i
    y : L j
    k : ι
    ⊢ Eq (Bracket.bracket (((DirectSum.of L i) x) k) (((DirectSum.of L j) y) k)) ( …
  -/
  obtain rfl | hik := Decidable.eq_or_ne i k
    /-
      case inl
      ι : Type v
      L : ι → Type w
      inst✝¹ : (i : ι) → LieRing (L i)
      inst✝ : DecidableEq ι
      i j : ι
      hij : Ne i j
      x : L i
      y : L j
      ⊢ Eq (Bracket.bracket (((DirectSum.of L i) x) i) (((DirectSum.of L j) y) i)) ( …
    -/
  · rw [of_eq_of_ne _ _ _ hij.symm, lie_zero, zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type v
      L : ι → Type w
      inst✝¹ : (i : ι) → LieRing (L i)
      inst✝ : DecidableEq ι
      i j : ι
      hij : Ne i j
      x : L i
      y : L j
      k : ι
      hik : Ne i k
      ⊢ Eq (Bracket.bracket (((DirectSum.of L i) x) k) (((DirectSum.of L j) y) k)) ( …
    -/
  · rw [of_eq_of_ne _ _ _ hik, zero_lie, zero_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem lie_of [DecidableEq ι] {i j : ι} (x : L i) (y : L j) :
    ⁅of L i x, of L j y⁆ = if hij : i = j then of L i ⁅x, hij.symm.recOn y⁆ else 0 := by
  /-
    ι : Type v
    L : ι → Type w
    inst✝¹ : (i : ι) → LieRing (L i)
    inst✝ : DecidableEq ι
    i j : ι
    x : L i
    y : L j
    ⊢ Eq (Bracket.bracket ((DirectSum.of L i) x) ((DirectSum.of L j) y)) (dite (Eq …
  -/
  obtain rfl | hij := Decidable.eq_or_ne i j
    /-
      case inl
      ι : Type v
      L : ι → Type w
      inst✝¹ : (i : ι) → LieRing (L i)
      inst✝ : DecidableEq ι
      i : ι
      x y : L i
      ⊢ Eq (Bracket.bracket ((DirectSum.of L i) x) ((DirectSum.of L i) y)) (dite (Eq …
    -/
  · simp only [lie_of_same L x y, dif_pos]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type v
      L : ι → Type w
      inst✝¹ : (i : ι) → LieRing (L i)
      inst✝ : DecidableEq ι
      i j : ι
      x : L i
      y : L j
      hij : Ne i j
      ⊢ Eq (Bracket.bracket ((DirectSum.of L i) x) ((DirectSum.of L j) y)) (dite (Eq …
    -/
  · simp only [lie_of_of_ne L hij x y, hij, dif_neg, dite_false]
    /-
      🎉 no goals
    -/


instance lieAlgebra : LieAlgebra R (⨁ i, L i) :=
  { (inferInstance : Module R _) with
    lie_smul := fun c x y => by
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        c : R
        x y : DirectSum ι fun i => L i
        ⊢ Eq (Bracket.bracket x (HSMul.hSMul c y)) (HSMul.hSMul c (Bracket.bracket x y))
      -/
      refine DFinsupp.ext fun _ => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext`
      /-
        R : Type u
        ι : Type v
        inst✝² : CommRing R
        L : ι → Type w
        inst✝¹ : (i : ι) → LieRing (L i)
        inst✝ : (i : ι) → LieAlgebra R (L i)
        c : R
        x y : DirectSum ι fun i => L i
        x✝ : ι
        ⊢ Eq ((Bracket.bracket x (HSMul.hSMul c y)) x✝) ((HSMul.hSMul c (Bracket.brack …
      -/
      simp only [zipWith_apply, smul_apply, bracket_apply, lie_smul] }
      /-
        🎉 no goals
      -/


/-- The inclusion of each component into the direct sum as morphism of Lie algebras. -/
@[simps]
def lieAlgebraOf [DecidableEq ι] (j : ι) : L j →ₗ⁅R⁆ ⨁ i, L i :=
  { lof R ι L j with
    toFun := of L j
    map_lie' := fun {x y} => by
      /-
        R : Type u
        ι : Type v
        inst✝³ : CommRing R
        L : ι → Type w
        inst✝² : (i : ι) → LieRing (L i)
        inst✝¹ : (i : ι) → LieAlgebra R (L i)
        inst✝ : DecidableEq ι
        j : ι
        x y : L j
        ⊢ Eq ({ toFun := ⇑(DirectSum.of L j), map_add' := ⋯, map_smul' := ⋯ }.toFun (B …
      -/
      refine DFinsupp.ext fun i => ?_ -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): Originally `ext i`
      /-
        R : Type u
        ι : Type v
        inst✝³ : CommRing R
        L : ι → Type w
        inst✝² : (i : ι) → LieRing (L i)
        inst✝¹ : (i : ι) → LieAlgebra R (L i)
        inst✝ : DecidableEq ι
        j : ι
        x y : L j
        i : ι
        ⊢ Eq (({ toFun := ⇑(DirectSum.of L j), map_add' := ⋯, map_smul' := ⋯ }.toFun ( …
      -/
      by_cases h : j = i
        /-
          case pos
          R : Type u
          ι : Type v
          inst✝³ : CommRing R
          L : ι → Type w
          inst✝² : (i : ι) → LieRing (L i)
          inst✝¹ : (i : ι) → LieAlgebra R (L i)
          inst✝ : DecidableEq ι
          j : ι
          x y : L j
          i : ι
          h : Eq j i
          ⊢ Eq (({ toFun := ⇑(DirectSum.of L j), map_add' := ⋯, map_smul' := ⋯ }.toFun ( …
        -/
      · rw [← h]
        -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
        -- with `simp [of, singleAddHom]`
        /-
          case pos
          R : Type u
          ι : Type v
          inst✝³ : CommRing R
          L : ι → Type w
          inst✝² : (i : ι) → LieRing (L i)
          inst✝¹ : (i : ι) → LieAlgebra R (L i)
          inst✝ : DecidableEq ι
          j : ι
          x y : L j
          i : ι
          h : Eq j i
          ⊢ Eq (({ toFun := ⇑(DirectSum.of L j), map_add' := ⋯, map_smul' := ⋯ }.toFun ( …
        -/
        simp only [of, singleAddHom, bracket_apply]
        /-
          case pos
          R : Type u
          ι : Type v
          inst✝³ : CommRing R
          L : ι → Type w
          inst✝² : (i : ι) → LieRing (L i)
          inst✝¹ : (i : ι) → LieAlgebra R (L i)
          inst✝ : DecidableEq ι
          j : ι
          x y : L j
          i : ι
          h : Eq j i
          ⊢ Eq (({ toFun := DFinsupp.single j, map_zero' := ⋯, map_add' := ⋯ } (Bracket. …
        -/
        erw [AddHom.coe_mk, single_apply, single_apply]
          /-
            case pos
            R : Type u
            ι : Type v
            inst✝³ : CommRing R
            L : ι → Type w
            inst✝² : (i : ι) → LieRing (L i)
            inst✝¹ : (i : ι) → LieAlgebra R (L i)
            inst✝ : DecidableEq ι
            j : ι
            x y : L j
            i : ι
            h : Eq j i
            ⊢ Eq (dite (Eq j j) (fun h => Eq.recOn h (Bracket.bracket x y)) fun h => 0) (B …
          -/
        · simp? [h] says simp only [h, ↓reduceDIte, single_apply]
          /-
            🎉 no goals
          -/
          /-
            case pos.hmul
            R : Type u
            ι : Type v
            inst✝³ : CommRing R
            L : ι → Type w
            inst✝² : (i : ι) → LieRing (L i)
            inst✝¹ : (i : ι) → LieAlgebra R (L i)
            inst✝ : DecidableEq ι
            j : ι
            x y : L j
            i : ι
            h : Eq j i
            ⊢ ∀ (x y : L j), Eq (DFinsupp.single j (HAdd.hAdd x y)) (HAdd.hAdd (DFinsupp.s …
          -/
        · intros
          /-
            case pos.hmul
            R : Type u
            ι : Type v
            inst✝³ : CommRing R
            L : ι → Type w
            inst✝² : (i : ι) → LieRing (L i)
            inst✝¹ : (i : ι) → LieAlgebra R (L i)
            inst✝ : DecidableEq ι
            j : ι
            x y : L j
            i : ι
            h : Eq j i
            x✝ y✝ : L j
            ⊢ Eq (DFinsupp.single j (HAdd.hAdd x✝ y✝)) (HAdd.hAdd (DFinsupp.single j x✝) ( …
          -/
          rw [single_add]
          /-
            🎉 no goals
          -/
      · -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
        -- with `simp [of, singleAddHom]`
        /-
          case neg
          R : Type u
          ι : Type v
          inst✝³ : CommRing R
          L : ι → Type w
          inst✝² : (i : ι) → LieRing (L i)
          inst✝¹ : (i : ι) → LieAlgebra R (L i)
          inst✝ : DecidableEq ι
          j : ι
          x y : L j
          i : ι
          h : Not (Eq j i)
          ⊢ Eq (({ toFun := ⇑(DirectSum.of L j), map_add' := ⋯, map_smul' := ⋯ }.toFun ( …
        -/
        simp only [of, singleAddHom, bracket_apply]
        /-
          case neg
          R : Type u
          ι : Type v
          inst✝³ : CommRing R
          L : ι → Type w
          inst✝² : (i : ι) → LieRing (L i)
          inst✝¹ : (i : ι) → LieAlgebra R (L i)
          inst✝ : DecidableEq ι
          j : ι
          x y : L j
          i : ι
          h : Not (Eq j i)
          ⊢ Eq (({ toFun := DFinsupp.single j, map_zero' := ⋯, map_add' := ⋯ } (Bracket. …
        -/
        erw [AddHom.coe_mk, single_apply, single_apply]
          /-
            case neg
            R : Type u
            ι : Type v
            inst✝³ : CommRing R
            L : ι → Type w
            inst✝² : (i : ι) → LieRing (L i)
            inst✝¹ : (i : ι) → LieAlgebra R (L i)
            inst✝ : DecidableEq ι
            j : ι
            x y : L j
            i : ι
            h : Not (Eq j i)
            ⊢ Eq (dite (Eq j i) (fun h => Eq.recOn h (Bracket.bracket x y)) fun h => 0) (B …
          -/
        · simp only [h, dite_false, single_apply, lie_self]
          /-
            🎉 no goals
          -/
          /-
            case neg.hmul
            R : Type u
            ι : Type v
            inst✝³ : CommRing R
            L : ι → Type w
            inst✝² : (i : ι) → LieRing (L i)
            inst✝¹ : (i : ι) → LieAlgebra R (L i)
            inst✝ : DecidableEq ι
            j : ι
            x y : L j
            i : ι
            h : Not (Eq j i)
            ⊢ ∀ (x y : L j), Eq (DFinsupp.single j (HAdd.hAdd x y)) (HAdd.hAdd (DFinsupp.s …
          -/
        · intros
          /-
            case neg.hmul
            R : Type u
            ι : Type v
            inst✝³ : CommRing R
            L : ι → Type w
            inst✝² : (i : ι) → LieRing (L i)
            inst✝¹ : (i : ι) → LieAlgebra R (L i)
            inst✝ : DecidableEq ι
            j : ι
            x y : L j
            i : ι
            h : Not (Eq j i)
            x✝ y✝ : L j
            ⊢ Eq (DFinsupp.single j (HAdd.hAdd x✝ y✝)) (HAdd.hAdd (DFinsupp.single j x✝) ( …
          -/
          rw [single_add] }
          /-
            🎉 no goals
          -/


/-- The projection map onto one component, as a morphism of Lie algebras. -/
@[simps]
def lieAlgebraComponent (j : ι) : (⨁ i, L i) →ₗ⁅R⁆ L j :=
  { component R ι L j with
    toFun := component R ι L j
                                /-
                                  R : Type u
                                  ι : Type v
                                  inst✝² : CommRing R
                                  L : ι → Type w
                                  inst✝¹ : (i : ι) → LieRing (L i)
                                  inst✝ : (i : ι) → LieAlgebra R (L i)
                                  j : ι
                                  x y : DirectSum ι fun i => L i
                                  ⊢ Eq ({ toFun := ⇑(DirectSum.component R ι L j), map_add' := ⋯, map_smul' := ⋯ …
                                -/
    map_lie' := fun {x y} => by simp [component, lapply] }
                                /-
                                  🎉 no goals
                                -/

-- Note(kmill): `ext` cannot generate an iff theorem here since `x` and `y` do not determine `R`.

@[ext (iff := false)]
theorem lieAlgebra_ext {x y : ⨁ i, L i}
    (h : ∀ i, lieAlgebraComponent R ι L i x = lieAlgebraComponent R ι L i y) : x = y :=
  DFinsupp.ext h


/-- Given a family of Lie algebras `L i`, together with a family of morphisms of Lie algebras
`f i : L i →ₗ⁅R⁆ L'` into a fixed Lie algebra `L'`, we have a natural linear map:
`(⨁ i, L i) →ₗ[R] L'`. If in addition `⁅f i x, f j y⁆ = 0` for any `x ∈ L i` and `y ∈ L j` (`i ≠ j`)
then this map is a morphism of Lie algebras. -/
@[simps]
def toLieAlgebra [DecidableEq ι] (L' : Type w₁) [LieRing L'] [LieAlgebra R L']
    (f : ∀ i, L i →ₗ⁅R⁆ L') (hf : Pairwise fun i j => ∀ (x : L i) (y : L j), ⁅f i x, f j y⁆ = 0) :
    (⨁ i, L i) →ₗ⁅R⁆ L' :=
  { toModule R ι L' fun i => (f i : L i →ₗ[R] L') with
    toFun := toModule R ι L' fun i => (f i : L i →ₗ[R] L')
    map_lie' := fun {x y} => by
      /-
        R : Type u
        ι : Type v
        inst✝⁵ : CommRing R
        L : ι → Type w
        inst✝⁴ : (i : ι) → LieRing (L i)
        inst✝³ : (i : ι) → LieAlgebra R (L i)
        inst✝² : DecidableEq ι
        L' : Type w₁
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        f : (i : ι) → LieHom R (L i) L'
        hf : Pairwise fun i j => ∀ (x : L i) (y : L j), Eq (Bracket.bracket ((f i) x)  …
        x y : DirectSum ι fun i => L i
        ⊢ Eq ({ toFun := ⇑(DirectSum.toModule R ι L' fun i => ↑(f i)), map_add' := ⋯,  …
      -/
      let f' i := (f i : L i →ₗ[R] L')
      /- The goal is linear in `y`. We can use this to reduce to the case that `y` has only one
        non-zero component. -/
      suffices ∀ (i : ι) (y : L i),
          toModule R ι L' f' ⁅x, of L i y⁆ =
            ⁅toModule R ι L' f' x, toModule R ι L' f' (of L i y)⁆ by
        simp only [← LieAlgebra.ad_apply R]
        rw [← LinearMap.comp_apply, ← LinearMap.comp_apply]
        congr; clear y; ext i y; exact this i y
      -- Similarly, we can reduce to the case that `x` has only one non-zero component.
      suffices ∀ (i j) (y : L i) (x : L j),
          toModule R ι L' f' ⁅of L j x, of L i y⁆ =
            ⁅toModule R ι L' f' (of L j x), toModule R ι L' f' (of L i y)⁆ by
        intro i y
        rw [← lie_skew x, ← lie_skew (toModule R ι L' f' x)]
        simp only [LinearMap.map_neg, neg_inj, ← LieAlgebra.ad_apply R]
        rw [← LinearMap.comp_apply, ← LinearMap.comp_apply]
        congr; clear x; ext j x; exact this j i x y
      /-
        R : Type u
        ι : Type v
        inst✝⁵ : CommRing R
        L : ι → Type w
        inst✝⁴ : (i : ι) → LieRing (L i)
        inst✝³ : (i : ι) → LieAlgebra R (L i)
        inst✝² : DecidableEq ι
        L' : Type w₁
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        f : (i : ι) → LieHom R (L i) L'
        hf : Pairwise fun i j => ∀ (x : L i) (y : L j), Eq (Bracket.bracket ((f i) x)  …
        x y : DirectSum ι fun i => L i
        f' : (i : ι) → LinearMap (RingHom.id R) (L i) L' := fun i => ↑(f i)
        ⊢ ∀ (i j : ι) (y : L i) (x : L j), Eq ((DirectSum.toModule R ι L' f') (Bracket …
      -/
      intro i j y x
      /-
        R : Type u
        ι : Type v
        inst✝⁵ : CommRing R
        L : ι → Type w
        inst✝⁴ : (i : ι) → LieRing (L i)
        inst✝³ : (i : ι) → LieAlgebra R (L i)
        inst✝² : DecidableEq ι
        L' : Type w₁
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        f : (i : ι) → LieHom R (L i) L'
        hf : Pairwise fun i j => ∀ (x : L i) (y : L j), Eq (Bracket.bracket ((f i) x)  …
        x✝ y✝ : DirectSum ι fun i => L i
        f' : (i : ι) → LinearMap (RingHom.id R) (L i) L' := fun i => ↑(f i)
        i j : ι
        y : L i
        x : L j
        ⊢ Eq ((DirectSum.toModule R ι L' f') (Bracket.bracket ((DirectSum.of L j) x) ( …
      -/
      simp only [f', coe_toModule_eq_coe_toAddMonoid, toAddMonoid_of]
      -- And finish with trivial case analysis.
      /-
        R : Type u
        ι : Type v
        inst✝⁵ : CommRing R
        L : ι → Type w
        inst✝⁴ : (i : ι) → LieRing (L i)
        inst✝³ : (i : ι) → LieAlgebra R (L i)
        inst✝² : DecidableEq ι
        L' : Type w₁
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        f : (i : ι) → LieHom R (L i) L'
        hf : Pairwise fun i j => ∀ (x : L i) (y : L j), Eq (Bracket.bracket ((f i) x)  …
        x✝ y✝ : DirectSum ι fun i => L i
        f' : (i : ι) → LinearMap (RingHom.id R) (L i) L' := fun i => ↑(f i)
        i j : ι
        y : L i
        x : L j
        ⊢ Eq ((DirectSum.toAddMonoid fun i => (↑(f i)).toAddMonoidHom) (Bracket.bracke …
      -/
      obtain rfl | hij := Decidable.eq_or_ne i j
      · simp_rw [lie_of_same, toAddMonoid_of, LinearMap.toAddMonoidHom_coe, LieHom.coe_toLinearMap,
          LieHom.map_lie]
      · simp_rw [lie_of_of_ne _ hij.symm, map_zero,  LinearMap.toAddMonoidHom_coe,
          LieHom.coe_toLinearMap, hf hij.symm x y] }


/-- The fact that this instance is necessary seems to be a bug in typeclass inference. See
[this Zulip thread](https://leanprover.zulipchat.com/#narrow/stream/113488-general/topic/
Typeclass.20resolution.20under.20binders/near/245151099). -/
instance lieRingOfIdeals : LieRing (⨁ i, I i) :=
  DirectSum.lieRing fun i => ↥(I i)


/-- See `DirectSum.lieRingOfIdeals` comment. -/
instance lieAlgebraOfIdeals : LieAlgebra R (⨁ i, I i) :=
  DirectSum.lieAlgebra fun i => ↥(I i)


