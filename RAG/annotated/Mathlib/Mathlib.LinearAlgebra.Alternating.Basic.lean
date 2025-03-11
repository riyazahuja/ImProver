/-- An alternating map from `ι → M` to `N`, denoted `M [⋀^ι]→ₗ[R] N`,
is a multilinear map that vanishes when two of its arguments are equal. -/
structure AlternatingMap extends MultilinearMap R (fun _ : ι => M) N where
  /-- The map is alternating: if `v` has two equal coordinates, then `f v = 0`. -/
  map_eq_zero_of_eq' : ∀ (v : ι → M) (i j : ι), v i = v j → i ≠ j → toFun v = 0


@[inherit_doc]
notation M " [⋀^" ι "]→ₗ[" R "] " N:100 => AlternatingMap R M N ι


instance instFunLike : FunLike (M [⋀^ι]→ₗ[R] N) (ι → M) N where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      inst✝¹⁰ : Semiring R
      M : Type u_2
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      N : Type u_3
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R N
      P : Type u_4
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : Module R P
      M' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      N' : Type u_6
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      f✝ f' : AlternatingMap R M N ι
      g✝ g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      f g : AlternatingMap R M N ι
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨_, _, _⟩, _⟩
    /-
      case mk.mk
      R : Type u_1
      inst✝¹⁰ : Semiring R
      M : Type u_2
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      N : Type u_3
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R N
      P : Type u_4
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : Module R P
      M' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      N' : Type u_6
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      f f' : AlternatingMap R M N ι
      g✝ g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      g : AlternatingMap R M N ι
      toFun✝ : (ι → M) → N
      map_update_add'✝ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (x y : M), Eq  …
      map_update_smul'✝ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (c : R) (x :  …
      map_eq_zero_of_eq'✝ : ∀ (v : ι → M) (i j : ι), Eq (v i) (v j) → Ne i j → Eq ({ …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, map_update_add' := map_update_ad …
      ⊢ Eq { toFun := toFun✝, map_update_add' := map_update_add'✝, map_update_smul'  …
    -/
    rcases g with ⟨⟨_, _, _⟩, _⟩
    /-
      case mk.mk.mk.mk
      R : Type u_1
      inst✝¹⁰ : Semiring R
      M : Type u_2
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : Module R M
      N : Type u_3
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R N
      P : Type u_4
      inst✝⁵ : AddCommMonoid P
      inst✝⁴ : Module R P
      M' : Type u_5
      inst✝³ : AddCommGroup M'
      inst✝² : Module R M'
      N' : Type u_6
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      f f' : AlternatingMap R M N ι
      g g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      toFun✝¹ : (ι → M) → N
      map_update_add'✝¹ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (x y : M), Eq …
      map_update_smul'✝¹ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (c : R) (x : …
      map_eq_zero_of_eq'✝¹ : ∀ (v : ι → M) (i j : ι), Eq (v i) (v j) → Ne i j → Eq ( …
      toFun✝ : (ι → M) → N
      map_update_add'✝ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (x y : M), Eq  …
      map_update_smul'✝ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (c : R) (x :  …
      map_eq_zero_of_eq'✝ : ∀ (v : ι → M) (i j : ι), Eq (v i) (v j) → Ne i j → Eq ({ …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, map_update_add' := map_update_a …
      ⊢ Eq { toFun := toFun✝¹, map_update_add' := map_update_add'✝¹, map_update_smul …
    -/
    congr
    /-
      🎉 no goals
    -/


@[simp]
theorem toFun_eq_coe : f.toFun = f :=
  rfl

-- Porting note: changed statement to reflect new `mk` signature

@[simp]
theorem coe_mk (f : MultilinearMap R (fun _ : ι => M) N) (h) :
    ⇑(⟨f, h⟩ : M [⋀^ι]→ₗ[R] N) = f :=
  rfl


protected theorem congr_fun {f g : M [⋀^ι]→ₗ[R] N} (h : f = g) (x : ι → M) : f x = g x :=
  congr_arg (fun h : M [⋀^ι]→ₗ[R] N => h x) h


protected theorem congr_arg (f : M [⋀^ι]→ₗ[R] N) {x y : ι → M} (h : x = y) : f x = f y :=
  congr_arg (fun x : ι → M => f x) h


theorem coe_injective : Injective ((↑) : M [⋀^ι]→ₗ[R] N → (ι → M) → N) :=
  DFunLike.coe_injective


@[norm_cast]
theorem coe_inj {f g : M [⋀^ι]→ₗ[R] N} : (f : (ι → M) → N) = g ↔ f = g :=
  coe_injective.eq_iff


@[ext]
theorem ext {f f' : M [⋀^ι]→ₗ[R] N} (H : ∀ x, f x = f' x) : f = f' :=
  DFunLike.ext _ _ H


instance coe : Coe (M [⋀^ι]→ₗ[R] N) (MultilinearMap R (fun _ : ι => M) N) :=
  ⟨fun x => x.toMultilinearMap⟩


@[simp, norm_cast]
theorem coe_multilinearMap : ⇑(f : MultilinearMap R (fun _ : ι => M) N) = f :=
  rfl


theorem coe_multilinearMap_injective :
    Function.Injective ((↑) : M [⋀^ι]→ₗ[R] N → MultilinearMap R (fun _ : ι => M) N) :=
  fun _ _ h => ext <| MultilinearMap.congr_fun h

-- Porting note: changed statement to reflect new `mk` signature.
-- Porting note: removed `simp`
-- @[simp]

theorem coe_multilinearMap_mk (f : (ι → M) → N) (h₁ h₂ h₃) :
    ((⟨⟨f, h₁, h₂⟩, h₃⟩ : M [⋀^ι]→ₗ[R] N) : MultilinearMap R (fun _ : ι => M) N) =
      ⟨f, @h₁, @h₂⟩ := by
  /-
    R : Type u_1
    inst✝⁴ : Semiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    f : (ι → M) → N
    h₁ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (x y : M), Eq (f (Function.u …
    h₂ : ∀ [inst : DecidableEq ι] (m : ι → M) (i : ι) (c : R) (x : M), Eq (f (Func …
    h₃ : ∀ (v : ι → M) (i j : ι), Eq (v i) (v j) → Ne i j → Eq ({ toFun := f, map_ …
    ⊢ Eq ↑{ toFun := f, map_update_add' := ⋯, map_update_smul' := ⋯, map_eq_zero_o …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_update_add [DecidableEq ι] (i : ι) (x y : M) :
    f (update v i (x + y)) = f (update v i x) + f (update v i y) :=
  f.map_update_add' v i x y


@[deprecated (since := "2024-11-03")] protected alias map_add := map_update_add


@[simp]
theorem map_update_sub [DecidableEq ι] (i : ι) (x y : M') :
    g' (update v' i (x - y)) = g' (update v' i x) - g' (update v' i y) :=
  g'.toMultilinearMap.map_update_sub v' i x y


@[deprecated (since := "2024-11-03")] protected alias map_sub := map_update_sub


@[simp]
theorem map_update_neg [DecidableEq ι] (i : ι) (x : M') :
    g' (update v' i (-x)) = -g' (update v' i x) :=
  g'.toMultilinearMap.map_update_neg v' i x


@[deprecated (since := "2024-11-03")] protected alias map_neg := map_update_neg


@[simp]
theorem map_update_smul [DecidableEq ι] (i : ι) (r : R) (x : M) :
    f (update v i (r • x)) = r • f (update v i x) :=
  f.map_update_smul' v i r x


@[deprecated (since := "2024-11-03")] protected alias map_smul := map_update_smul


@[simp]
theorem map_eq_zero_of_eq (v : ι → M) {i j : ι} (h : v i = v j) (hij : i ≠ j) : f v = 0 :=
  f.map_eq_zero_of_eq' v i j h hij


theorem map_coord_zero {m : ι → M} (i : ι) (h : m i = 0) : f m = 0 :=
  f.toMultilinearMap.map_coord_zero i h


@[simp]
theorem map_update_zero [DecidableEq ι] (m : ι → M) (i : ι) : f (update m i 0) = 0 :=
  f.toMultilinearMap.map_update_zero m i


@[simp]
theorem map_zero [Nonempty ι] : f 0 = 0 :=
  f.toMultilinearMap.map_zero


theorem map_eq_zero_of_not_injective (v : ι → M) (hv : ¬Function.Injective v) : f v = 0 := by
  /-
    R : Type u_1
    inst✝⁴ : Semiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    hv : Not (Function.Injective v)
    ⊢ Eq (f v) 0
  -/
  rw [Function.Injective] at hv
  /-
    R : Type u_1
    inst✝⁴ : Semiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    hv : Not (∀ ⦃a₁ a₂ : ι⦄, Eq (v a₁) (v a₂) → Eq a₁ a₂)
    ⊢ Eq (f v) 0
  -/
  push_neg at hv
  /-
    R : Type u_1
    inst✝⁴ : Semiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    hv : Exists fun ⦃a₁⦄ => Exists fun ⦃a₂⦄ => And (Eq (v a₁) (v a₂)) (Ne a₁ a₂)
    ⊢ Eq (f v) 0
  -/
  rcases hv with ⟨i₁, i₂, heq, hne⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : Semiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    i₁ i₂ : ι
    heq : Eq (v i₁) (v i₂)
    hne : Ne i₁ i₂
    ⊢ Eq (f v) 0
  -/
  exact f.map_eq_zero_of_eq v heq hne
  /-
    🎉 no goals
  -/


instance smul : SMul S (M [⋀^ι]→ₗ[R] N) :=
  ⟨fun c f =>
    { c • (f : MultilinearMap R (fun _ : ι => M) N) with
                                                  /-
                                                    R : Type u_1
                                                    inst✝¹³ : Semiring R
                                                    M : Type u_2
                                                    inst✝¹² : AddCommMonoid M
                                                    inst✝¹¹ : Module R M
                                                    N : Type u_3
                                                    inst✝¹⁰ : AddCommMonoid N
                                                    inst✝⁹ : Module R N
                                                    P : Type u_4
                                                    inst✝⁸ : AddCommMonoid P
                                                    inst✝⁷ : Module R P
                                                    M' : Type u_5
                                                    inst✝⁶ : AddCommGroup M'
                                                    inst✝⁵ : Module R M'
                                                    N' : Type u_6
                                                    inst✝⁴ : AddCommGroup N'
                                                    inst✝³ : Module R N'
                                                    ι : Type u_7
                                                    ι' : Type u_8
                                                    ι'' : Type u_9
                                                    f✝ f' : AlternatingMap R M N ι
                                                    g g₂ : AlternatingMap R M N' ι
                                                    g' : AlternatingMap R M' N' ι
                                                    v✝ : ι → M
                                                    v' : ι → M'
                                                    S : Type u_10
                                                    inst✝² : Monoid S
                                                    inst✝¹ : DistribMulAction S N
                                                    inst✝ : SMulCommClass R S N
                                                    c : S
                                                    f : AlternatingMap R M N ι
                                                    v : ι → M
                                                    i j : ι
                                                    h : Eq (v i) (v j)
                                                    hij : Ne i j
                                                    ⊢ Eq (__src✝.toFun v) 0
                                                  -/
      map_eq_zero_of_eq' := fun v i j h hij => by simp [f.map_eq_zero_of_eq v h hij] }⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem smul_apply (c : S) (m : ι → M) : (c • f) m = c • f m :=
  rfl


@[norm_cast]
theorem coe_smul (c : S) : ↑(c • f) = c • (f : MultilinearMap R (fun _ : ι => M) N) :=
  rfl


theorem coeFn_smul (c : S) (f : M [⋀^ι]→ₗ[R] N) : ⇑(c • f) = c • ⇑f :=
  rfl


instance isCentralScalar [DistribMulAction Sᵐᵒᵖ N] [IsCentralScalar S N] :
    IsCentralScalar S (M [⋀^ι]→ₗ[R] N) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


/-- The cartesian product of two alternating maps, as an alternating map. -/
@[simps!]
def prod (f : M [⋀^ι]→ₗ[R] N) (g : M [⋀^ι]→ₗ[R] P) : M [⋀^ι]→ₗ[R] (N × P) :=
  { f.toMultilinearMap.prod g.toMultilinearMap with
    map_eq_zero_of_eq' := fun _ _ _ h hne =>
      Prod.ext (f.map_eq_zero_of_eq _ h hne) (g.map_eq_zero_of_eq _ h hne) }


@[simp]
theorem coe_prod (f : M [⋀^ι]→ₗ[R] N) (g : M [⋀^ι]→ₗ[R] P) :
    (f.prod g : MultilinearMap R (fun _ : ι => M) (N × P)) = MultilinearMap.prod f g :=
  rfl


/-- Combine a family of alternating maps with the same domain and codomains `N i` into an
alternating map taking values in the space of functions `Π i, N i`. -/
@[simps!]
def pi {ι' : Type*} {N : ι' → Type*} [∀ i, AddCommMonoid (N i)] [∀ i, Module R (N i)]
    (f : ∀ i, M [⋀^ι]→ₗ[R] N i) : M [⋀^ι]→ₗ[R] (∀ i, N i) :=
  { MultilinearMap.pi fun a => (f a).toMultilinearMap with
    map_eq_zero_of_eq' := fun _ _ _ h hne => funext fun a => (f a).map_eq_zero_of_eq _ h hne }


@[simp]
theorem coe_pi {ι' : Type*} {N : ι' → Type*} [∀ i, AddCommMonoid (N i)] [∀ i, Module R (N i)]
    (f : ∀ i, M [⋀^ι]→ₗ[R] N i) :
    (pi f : MultilinearMap R (fun _ : ι => M) (∀ i, N i)) = MultilinearMap.pi fun a => f a :=
  rfl


/-- Given an alternating `R`-multilinear map `f` taking values in `R`, `f.smul_right z` is the map
sending `m` to `f m • z`. -/
@[simps!]
def smulRight {R M₁ M₂ ι : Type*} [CommSemiring R] [AddCommMonoid M₁] [AddCommMonoid M₂]
    [Module R M₁] [Module R M₂] (f : M₁ [⋀^ι]→ₗ[R] R) (z : M₂) : M₁ [⋀^ι]→ₗ[R] M₂ :=
  { f.toMultilinearMap.smulRight z with
                                                /-
                                                  R✝ : Type u_1
                                                  inst✝¹⁵ : Semiring R✝
                                                  M : Type u_2
                                                  inst✝¹⁴ : AddCommMonoid M
                                                  inst✝¹³ : Module R✝ M
                                                  N : Type u_3
                                                  inst✝¹² : AddCommMonoid N
                                                  inst✝¹¹ : Module R✝ N
                                                  P : Type u_4
                                                  inst✝¹⁰ : AddCommMonoid P
                                                  inst✝⁹ : Module R✝ P
                                                  M' : Type u_5
                                                  inst✝⁸ : AddCommGroup M'
                                                  inst✝⁷ : Module R✝ M'
                                                  N' : Type u_6
                                                  inst✝⁶ : AddCommGroup N'
                                                  inst✝⁵ : Module R✝ N'
                                                  ι✝ : Type u_7
                                                  ι' : Type u_8
                                                  ι'' : Type u_9
                                                  f✝ f' : AlternatingMap R✝ M N ι✝
                                                  g g₂ : AlternatingMap R✝ M N' ι✝
                                                  g' : AlternatingMap R✝ M' N' ι✝
                                                  v✝ : ι✝ → M
                                                  v' : ι✝ → M'
                                                  R : Type u_10
                                                  M₁ : Type u_11
                                                  M₂ : Type u_12
                                                  ι : Type u_13
                                                  inst✝⁴ : CommSemiring R
                                                  inst✝³ : AddCommMonoid M₁
                                                  inst✝² : AddCommMonoid M₂
                                                  inst✝¹ : Module R M₁
                                                  inst✝ : Module R M₂
                                                  f : AlternatingMap R M₁ R ι
                                                  z : M₂
                                                  v : ι → M₁
                                                  i j : ι
                                                  h : Eq (v i) (v j)
                                                  hne : Ne i j
                                                  ⊢ Eq (__src✝.toFun v) 0
                                                -/
    map_eq_zero_of_eq' := fun v i j h hne => by simp [f.map_eq_zero_of_eq v h hne] }
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem coe_smulRight {R M₁ M₂ ι : Type*} [CommSemiring R] [AddCommMonoid M₁] [AddCommMonoid M₂]
    [Module R M₁] [Module R M₂] (f : M₁ [⋀^ι]→ₗ[R] R) (z : M₂) :
    (f.smulRight z : MultilinearMap R (fun _ : ι => M₁) M₂) = MultilinearMap.smulRight f z :=
  rfl


instance add : Add (M [⋀^ι]→ₗ[R] N) :=
  ⟨fun a b =>
    { (a + b : MultilinearMap R (fun _ : ι => M) N) with
      map_eq_zero_of_eq' := fun v i j h hij => by
        /-
          R : Type u_1
          inst✝¹⁰ : Semiring R
          M : Type u_2
          inst✝⁹ : AddCommMonoid M
          inst✝⁸ : Module R M
          N : Type u_3
          inst✝⁷ : AddCommMonoid N
          inst✝⁶ : Module R N
          P : Type u_4
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          M' : Type u_5
          inst✝³ : AddCommGroup M'
          inst✝² : Module R M'
          N' : Type u_6
          inst✝¹ : AddCommGroup N'
          inst✝ : Module R N'
          ι : Type u_7
          ι' : Type u_8
          ι'' : Type u_9
          f f' : AlternatingMap R M N ι
          g g₂ : AlternatingMap R M N' ι
          g' : AlternatingMap R M' N' ι
          v✝ : ι → M
          v' : ι → M'
          a b : AlternatingMap R M N ι
          v : ι → M
          i j : ι
          h : Eq (v i) (v j)
          hij : Ne i j
          ⊢ Eq (__src✝.toFun v) 0
        -/
        simp [a.map_eq_zero_of_eq v h hij, b.map_eq_zero_of_eq v h hij] }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem add_apply : (f + f') v = f v + f' v :=
  rfl


@[norm_cast]
theorem coe_add : (↑(f + f') : MultilinearMap R (fun _ : ι => M) N) = f + f' :=
  rfl


instance zero : Zero (M [⋀^ι]→ₗ[R] N) :=
  ⟨{ (0 : MultilinearMap R (fun _ : ι => M) N) with
                                                /-
                                                  R : Type u_1
                                                  inst✝¹⁰ : Semiring R
                                                  M : Type u_2
                                                  inst✝⁹ : AddCommMonoid M
                                                  inst✝⁸ : Module R M
                                                  N : Type u_3
                                                  inst✝⁷ : AddCommMonoid N
                                                  inst✝⁶ : Module R N
                                                  P : Type u_4
                                                  inst✝⁵ : AddCommMonoid P
                                                  inst✝⁴ : Module R P
                                                  M' : Type u_5
                                                  inst✝³ : AddCommGroup M'
                                                  inst✝² : Module R M'
                                                  N' : Type u_6
                                                  inst✝¹ : AddCommGroup N'
                                                  inst✝ : Module R N'
                                                  ι : Type u_7
                                                  ι' : Type u_8
                                                  ι'' : Type u_9
                                                  f f' : AlternatingMap R M N ι
                                                  g g₂ : AlternatingMap R M N' ι
                                                  g' : AlternatingMap R M' N' ι
                                                  v : ι → M
                                                  v' : ι → M'
                                                  x✝⁴ : ι → M
                                                  x✝³ x✝² : ι
                                                  x✝¹ : Eq (x✝⁴ x✝³) (x✝⁴ x✝²)
                                                  x✝ : Ne x✝³ x✝²
                                                  ⊢ Eq (__src✝.toFun x✝⁴) 0
                                                -/
      map_eq_zero_of_eq' := fun _ _ _ _ _ => by simp }⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem zero_apply : (0 : M [⋀^ι]→ₗ[R] N) v = 0 :=
  rfl


@[norm_cast]
theorem coe_zero : ((0 : M [⋀^ι]→ₗ[R] N) : MultilinearMap R (fun _ : ι => M) N) = 0 :=
  rfl


@[simp]
theorem mk_zero :
    mk (0 : MultilinearMap R (fun _ : ι ↦ M) N) (0 : M [⋀^ι]→ₗ[R] N).2 = 0 :=
  rfl


instance inhabited : Inhabited (M [⋀^ι]→ₗ[R] N) :=
  ⟨0⟩


instance addCommMonoid : AddCommMonoid (M [⋀^ι]→ₗ[R] N) :=
  coe_injective.addCommMonoid _ rfl (fun _ _ => rfl) fun _ _ => coeFn_smul _ _


instance neg : Neg (M [⋀^ι]→ₗ[R] N') :=
  ⟨fun f =>
    { -(f : MultilinearMap R (fun _ : ι => M) N') with
                                                  /-
                                                    R : Type u_1
                                                    inst✝¹⁰ : Semiring R
                                                    M : Type u_2
                                                    inst✝⁹ : AddCommMonoid M
                                                    inst✝⁸ : Module R M
                                                    N : Type u_3
                                                    inst✝⁷ : AddCommMonoid N
                                                    inst✝⁶ : Module R N
                                                    P : Type u_4
                                                    inst✝⁵ : AddCommMonoid P
                                                    inst✝⁴ : Module R P
                                                    M' : Type u_5
                                                    inst✝³ : AddCommGroup M'
                                                    inst✝² : Module R M'
                                                    N' : Type u_6
                                                    inst✝¹ : AddCommGroup N'
                                                    inst✝ : Module R N'
                                                    ι : Type u_7
                                                    ι' : Type u_8
                                                    ι'' : Type u_9
                                                    f✝ f' : AlternatingMap R M N ι
                                                    g g₂ : AlternatingMap R M N' ι
                                                    g' : AlternatingMap R M' N' ι
                                                    v✝ : ι → M
                                                    v' : ι → M'
                                                    f : AlternatingMap R M N' ι
                                                    v : ι → M
                                                    i j : ι
                                                    h : Eq (v i) (v j)
                                                    hij : Ne i j
                                                    ⊢ Eq (__src✝.toFun v) 0
                                                  -/
      map_eq_zero_of_eq' := fun v i j h hij => by simp [f.map_eq_zero_of_eq v h hij] }⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem neg_apply (m : ι → M) : (-g) m = -g m :=
  rfl


@[norm_cast]
theorem coe_neg : ((-g : M [⋀^ι]→ₗ[R] N') : MultilinearMap R (fun _ : ι => M) N') = -g :=
  rfl


instance sub : Sub (M [⋀^ι]→ₗ[R] N') :=
  ⟨fun f g =>
    { (f - g : MultilinearMap R (fun _ : ι => M) N') with
      map_eq_zero_of_eq' := fun v i j h hij => by
        /-
          R : Type u_1
          inst✝¹⁰ : Semiring R
          M : Type u_2
          inst✝⁹ : AddCommMonoid M
          inst✝⁸ : Module R M
          N : Type u_3
          inst✝⁷ : AddCommMonoid N
          inst✝⁶ : Module R N
          P : Type u_4
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          M' : Type u_5
          inst✝³ : AddCommGroup M'
          inst✝² : Module R M'
          N' : Type u_6
          inst✝¹ : AddCommGroup N'
          inst✝ : Module R N'
          ι : Type u_7
          ι' : Type u_8
          ι'' : Type u_9
          f✝ f' : AlternatingMap R M N ι
          g✝ g₂ : AlternatingMap R M N' ι
          g' : AlternatingMap R M' N' ι
          v✝ : ι → M
          v' : ι → M'
          f g : AlternatingMap R M N' ι
          v : ι → M
          i j : ι
          h : Eq (v i) (v j)
          hij : Ne i j
          ⊢ Eq (__src✝.toFun v) 0
        -/
        simp [f.map_eq_zero_of_eq v h hij, g.map_eq_zero_of_eq v h hij] }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem sub_apply (m : ι → M) : (g - g₂) m = g m - g₂ m :=
  rfl


@[norm_cast]
theorem coe_sub : (↑(g - g₂) : MultilinearMap R (fun _ : ι => M) N') = g - g₂ :=
  rfl


instance addCommGroup : AddCommGroup (M [⋀^ι]→ₗ[R] N') :=
  coe_injective.addCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => coeFn_smul _ _) fun _ _ => coeFn_smul _ _

instance distribMulAction : DistribMulAction S (M [⋀^ι]→ₗ[R] N) where
  one_smul _ := ext fun _ => one_smul _ _
  mul_smul _ _ _ := ext fun _ => mul_smul _ _ _
  smul_zero _ := ext fun _ => smul_zero _
  smul_add _ _ _ := ext fun _ => smul_add _ _ _


/-- The space of multilinear maps over an algebra over `R` is a module over `R`, for the pointwise
addition and scalar multiplication. -/
instance module : Module S (M [⋀^ι]→ₗ[R] N) where
  add_smul _ _ _ := ext fun _ => add_smul _ _ _
  zero_smul _ := ext fun _ => zero_smul _ _


instance noZeroSMulDivisors [NoZeroSMulDivisors S N] :
    NoZeroSMulDivisors S (M [⋀^ι]→ₗ[R] N) :=
  coe_injective.noZeroSMulDivisors _ rfl coeFn_smul


/-- The natural equivalence between linear maps from `M` to `N`
and `1`-multilinear alternating maps from `M` to `N`. -/
@[simps!]
def ofSubsingleton [Subsingleton ι] (i : ι) : (M →ₗ[R] N) ≃ (M [⋀^ι]→ₗ[R] N) where
  toFun f := ⟨MultilinearMap.ofSubsingleton R M N i f, fun _ _ _ _ ↦ absurd (Subsingleton.elim _ _)⟩
  invFun f := (MultilinearMap.ofSubsingleton R M N i).symm f
  left_inv _ := rfl
  right_inv _ := coe_multilinearMap_injective <|
    (MultilinearMap.ofSubsingleton R M N i).apply_symm_apply _


/-- The constant map is alternating when `ι` is empty. -/
@[simps (config := .asFn)]
def constOfIsEmpty [IsEmpty ι] (m : N) : M [⋀^ι]→ₗ[R] N :=
  { MultilinearMap.constOfIsEmpty R _ m with
    toFun := Function.const _ m
    map_eq_zero_of_eq' := fun _ => isEmptyElim }


/-- Restrict the codomain of an alternating map to a submodule. -/
@[simps]
def codRestrict (f : M [⋀^ι]→ₗ[R] N) (p : Submodule R N) (h : ∀ v, f v ∈ p) :
    M [⋀^ι]→ₗ[R] p :=
  { f.toMultilinearMap.codRestrict p h with
    toFun := fun v => ⟨f v, h v⟩
    map_eq_zero_of_eq' := fun _ _ _ hv hij => Subtype.ext <| map_eq_zero_of_eq _ _ hv hij }


/-- Composing an alternating map with a linear map on the left gives again an alternating map. -/
def compAlternatingMap (g : N →ₗ[R] N₂) (f : M [⋀^ι]→ₗ[R] N) : M [⋀^ι]→ₗ[R] N₂ where
  __ := g.compMultilinearMap (f : MultilinearMap R (fun _ : ι => M) N)
                                       /-
                                         R : Type u_1
                                         inst✝¹² : Semiring R
                                         M : Type u_2
                                         inst✝¹¹ : AddCommMonoid M
                                         inst✝¹⁰ : Module R M
                                         N : Type u_3
                                         inst✝⁹ : AddCommMonoid N
                                         inst✝⁸ : Module R N
                                         P : Type u_4
                                         inst✝⁷ : AddCommMonoid P
                                         inst✝⁶ : Module R P
                                         M' : Type u_5
                                         inst✝⁵ : AddCommGroup M'
                                         inst✝⁴ : Module R M'
                                         N' : Type u_6
                                         inst✝³ : AddCommGroup N'
                                         inst✝² : Module R N'
                                         ι : Type u_7
                                         ι' : Type u_8
                                         ι'' : Type u_9
                                         S : Type u_10
                                         N₂ : Type u_11
                                         inst✝¹ : AddCommMonoid N₂
                                         inst✝ : Module R N₂
                                         g : LinearMap (RingHom.id R) N N₂
                                         f : AlternatingMap R M N ι
                                         v : ι → M
                                         i j : ι
                                         h : Eq (v i) (v j)
                                         hij : Ne i j
                                         ⊢ Eq (__spread✝⁻⁰.toFun v) 0
                                       -/
  map_eq_zero_of_eq' v i j h hij := by simp [f.map_eq_zero_of_eq v h hij]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem coe_compAlternatingMap (g : N →ₗ[R] N₂) (f : M [⋀^ι]→ₗ[R] N) :
    ⇑(g.compAlternatingMap f) = g ∘ f :=
  rfl


@[simp]
theorem compAlternatingMap_apply (g : N →ₗ[R] N₂) (f : M [⋀^ι]→ₗ[R] N) (m : ι → M) :
    g.compAlternatingMap f m = g (f m) :=
  rfl


@[simp]
theorem compAlternatingMap_zero (g : N →ₗ[R] N₂) :
    g.compAlternatingMap (0 : M [⋀^ι]→ₗ[R] N) = 0 :=
  AlternatingMap.ext fun _ => map_zero g


@[simp]
theorem zero_compAlternatingMap (f: M [⋀^ι]→ₗ[R] N) :
    (0 : N →ₗ[R] N₂).compAlternatingMap f = 0 := rfl


@[simp]
theorem compAlternatingMap_add (g : N →ₗ[R] N₂) (f₁ f₂ : M [⋀^ι]→ₗ[R] N) :
    g.compAlternatingMap (f₁ + f₂) = g.compAlternatingMap f₁ + g.compAlternatingMap f₂ :=
  AlternatingMap.ext fun _ => map_add g _ _


@[simp]
theorem add_compAlternatingMap (g₁ g₂ : N →ₗ[R] N₂) (f: M [⋀^ι]→ₗ[R] N) :
    (g₁ + g₂).compAlternatingMap f = g₁.compAlternatingMap f + g₂.compAlternatingMap f := rfl


@[simp]
theorem compAlternatingMap_smul [Monoid S] [DistribMulAction S N] [DistribMulAction S N₂]
    [SMulCommClass R S N] [SMulCommClass R S N₂] [CompatibleSMul N N₂ S R]
    (g : N →ₗ[R] N₂) (s : S) (f : M [⋀^ι]→ₗ[R] N) :
    g.compAlternatingMap (s • f) = s • g.compAlternatingMap f :=
  AlternatingMap.ext fun _ => g.map_smul_of_tower _ _


@[simp]
theorem smul_compAlternatingMap [Monoid S] [DistribMulAction S N₂] [SMulCommClass R S N₂]
    (g : N →ₗ[R] N₂) (s : S) (f : M [⋀^ι]→ₗ[R] N) :
    (s • g).compAlternatingMap f = s • g.compAlternatingMap f := rfl


variable (S) in
/-- `LinearMap.compAlternatingMap` as an `S`-linear map. -/
@[simps]
def compAlternatingMapₗ [Semiring S] [Module S N] [Module S N₂]
    [SMulCommClass R S N] [SMulCommClass R S N₂] [LinearMap.CompatibleSMul N N₂ S R]
    (g : N →ₗ[R] N₂) :
    (M [⋀^ι]→ₗ[R] N) →ₗ[S] (M [⋀^ι]→ₗ[R] N₂) where
  toFun := g.compAlternatingMap
  map_add' := g.compAlternatingMap_add
  map_smul' := g.compAlternatingMap_smul


theorem smulRight_eq_comp {R M₁ M₂ ι : Type*} [CommSemiring R] [AddCommMonoid M₁]
    [AddCommMonoid M₂] [Module R M₁] [Module R M₂] (f : M₁ [⋀^ι]→ₗ[R] R) (z : M₂) :
    f.smulRight z = (LinearMap.id.smulRight z).compAlternatingMap f :=
  rfl


@[simp]
theorem subtype_compAlternatingMap_codRestrict (f : M [⋀^ι]→ₗ[R] N) (p : Submodule R N)
    (h) : p.subtype.compAlternatingMap (f.codRestrict p h) = f :=
  AlternatingMap.ext fun _ => rfl


@[simp]
theorem compAlternatingMap_codRestrict (g : N →ₗ[R] N₂) (f : M [⋀^ι]→ₗ[R] N)
    (p : Submodule R N₂) (h) :
    (g.codRestrict p h).compAlternatingMap f =
      (g.compAlternatingMap f).codRestrict p fun v => h (f v) :=
  AlternatingMap.ext fun _ => rfl


/-- Composing an alternating map with the same linear map on each argument gives again an
alternating map. -/
def compLinearMap (f : M [⋀^ι]→ₗ[R] N) (g : M₂ →ₗ[R] M) : M₂ [⋀^ι]→ₗ[R] N :=
  { (f : MultilinearMap R (fun _ : ι => M) N).compLinearMap fun _ => g with
    map_eq_zero_of_eq' := fun _ _ _ h hij => f.map_eq_zero_of_eq _ (LinearMap.congr_arg h) hij }


theorem coe_compLinearMap (f : M [⋀^ι]→ₗ[R] N) (g : M₂ →ₗ[R] M) :
    ⇑(f.compLinearMap g) = f ∘ (g ∘ ·) :=
  rfl


@[simp]
theorem compLinearMap_apply (f : M [⋀^ι]→ₗ[R] N) (g : M₂ →ₗ[R] M) (v : ι → M₂) :
    f.compLinearMap g v = f fun i => g (v i) :=
  rfl


/-- Composing an alternating map twice with the same linear map in each argument is
the same as composing with their composition. -/
theorem compLinearMap_assoc (f : M [⋀^ι]→ₗ[R] N) (g₁ : M₂ →ₗ[R] M) (g₂ : M₃ →ₗ[R] M₂) :
    (f.compLinearMap g₁).compLinearMap g₂ = f.compLinearMap (g₁ ∘ₗ g₂) :=
  rfl


@[simp]
theorem zero_compLinearMap (g : M₂ →ₗ[R] M) : (0 : M [⋀^ι]→ₗ[R] N).compLinearMap g = 0 := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    ι : Type u_7
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    g : LinearMap (RingHom.id R) M₂ M
    ⊢ Eq (AlternatingMap.compLinearMap 0 g) 0
  -/
  ext
  /-
    case H
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    ι : Type u_7
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    g : LinearMap (RingHom.id R) M₂ M
    x✝ : ι → M₂
    ⊢ Eq ((AlternatingMap.compLinearMap 0 g) x✝) (0 x✝)
  -/
  simp only [compLinearMap_apply, zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_compLinearMap (f₁ f₂ : M [⋀^ι]→ₗ[R] N) (g : M₂ →ₗ[R] M) :
    (f₁ + f₂).compLinearMap g = f₁.compLinearMap g + f₂.compLinearMap g := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    ι : Type u_7
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f₁ f₂ : AlternatingMap R M N ι
    g : LinearMap (RingHom.id R) M₂ M
    ⊢ Eq ((HAdd.hAdd f₁ f₂).compLinearMap g) (HAdd.hAdd (f₁.compLinearMap g) (f₂.c …
  -/
  ext
  /-
    case H
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    ι : Type u_7
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f₁ f₂ : AlternatingMap R M N ι
    g : LinearMap (RingHom.id R) M₂ M
    x✝ : ι → M₂
    ⊢ Eq (((HAdd.hAdd f₁ f₂).compLinearMap g) x✝) ((HAdd.hAdd (f₁.compLinearMap g) …
  -/
  simp only [compLinearMap_apply, add_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem compLinearMap_zero [Nonempty ι] (f : M [⋀^ι]→ₗ[R] N) :
    f.compLinearMap (0 : M₂ →ₗ[R] M) = 0 := by
  /-
    R : Type u_1
    inst✝⁷ : Semiring R
    M : Type u_2
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    N : Type u_3
    inst✝⁴ : AddCommMonoid N
    inst✝³ : Module R N
    ι : Type u_7
    M₂ : Type u_10
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Nonempty ι
    f : AlternatingMap R M N ι
    ⊢ Eq (f.compLinearMap 0) 0
  -/
  ext
  /-
    case H
    R : Type u_1
    inst✝⁷ : Semiring R
    M : Type u_2
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    N : Type u_3
    inst✝⁴ : AddCommMonoid N
    inst✝³ : Module R N
    ι : Type u_7
    M₂ : Type u_10
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M₂
    inst✝ : Nonempty ι
    f : AlternatingMap R M N ι
    x✝ : ι → M₂
    ⊢ Eq ((f.compLinearMap 0) x✝) (0 x✝)
  -/
  simp_rw [compLinearMap_apply, LinearMap.zero_apply, ← Pi.zero_def, map_zero, zero_apply]
  /-
    🎉 no goals
  -/


/-- Composing an alternating map with the identity linear map in each argument. -/
@[simp]
theorem compLinearMap_id (f : M [⋀^ι]→ₗ[R] N) : f.compLinearMap LinearMap.id = f :=
  ext fun _ => rfl


/-- Composing with a surjective linear map is injective. -/
theorem compLinearMap_injective (f : M₂ →ₗ[R] M) (hf : Function.Surjective f) :
    Function.Injective fun g : M [⋀^ι]→ₗ[R] N => g.compLinearMap f := fun g₁ g₂ h =>
  ext fun x => by
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      ι : Type u_7
      M₂ : Type u_10
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      f : LinearMap (RingHom.id R) M₂ M
      hf : Function.Surjective ⇑f
      g₁ g₂ : AlternatingMap R M N ι
      h : Eq ((fun g => g.compLinearMap f) g₁) ((fun g => g.compLinearMap f) g₂)
      x : ι → M
      ⊢ Eq (g₁ x) (g₂ x)
    -/
    simpa [Function.surjInv_eq hf] using AlternatingMap.ext_iff.mp h (Function.surjInv hf ∘ x)
    /-
      🎉 no goals
    -/


theorem compLinearMap_inj (f : M₂ →ₗ[R] M) (hf : Function.Surjective f)
    (g₁ g₂ : M [⋀^ι]→ₗ[R] N) : g₁.compLinearMap f = g₂.compLinearMap f ↔ g₁ = g₂ :=
  (compLinearMap_injective _ hf).eq_iff


/-- Construct a linear equivalence between maps from a linear equivalence between domains. -/
@[simps apply]
def domLCongr (e : M ≃ₗ[R] M₂) : M [⋀^ι]→ₗ[R] N ≃ₗ[S] (M₂ [⋀^ι]→ₗ[R] N) where
  toFun f := f.compLinearMap e.symm
  invFun g := g.compLinearMap e
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv f := AlternatingMap.ext fun _ => f.congr_arg <| funext fun _ => e.symm_apply_apply _
  right_inv f := AlternatingMap.ext fun _ => f.congr_arg <| funext fun _ => e.apply_symm_apply _


@[simp]
theorem domLCongr_refl : domLCongr R N ι S (LinearEquiv.refl R M) = LinearEquiv.refl S _ :=
  LinearEquiv.ext fun _ => AlternatingMap.ext fun _ => rfl


@[simp]
theorem domLCongr_symm (e : M ≃ₗ[R] M₂) : (domLCongr R N ι S e).symm = domLCongr R N ι S e.symm :=
  rfl


theorem domLCongr_trans (e : M ≃ₗ[R] M₂) (f : M₂ ≃ₗ[R] M₃) :
    (domLCongr R N ι S e).trans (domLCongr R N ι S f) = domLCongr R N ι S (e.trans f) :=
  rfl


/-- Composing an alternating map with the same linear equiv on each argument gives the zero map
if and only if the alternating map is the zero map. -/
@[simp]
theorem compLinearEquiv_eq_zero_iff (f : M [⋀^ι]→ₗ[R] N) (g : M₂ ≃ₗ[R] M) :
    f.compLinearMap (g : M₂ →ₗ[R] M) = 0 ↔ f = 0 :=
  (domLCongr R N ι ℕ g.symm).map_eq_zero_iff


theorem map_update_sum {α : Type*} [DecidableEq ι] (t : Finset α) (i : ι) (g : α → M) (m : ι → M) :
    f (update m i (∑ a ∈ t, g a)) = ∑ a ∈ t, f (update m i (g a)) :=
  f.toMultilinearMap.map_update_sum t i g m


theorem map_update_self [DecidableEq ι] {i j : ι} (hij : i ≠ j) :
    f (Function.update v i (v j)) = 0 :=
                            /-
                              R : Type u_1
                              inst✝⁵ : Semiring R
                              M : Type u_2
                              inst✝⁴ : AddCommMonoid M
                              inst✝³ : Module R M
                              N : Type u_3
                              inst✝² : AddCommMonoid N
                              inst✝¹ : Module R N
                              ι : Type u_7
                              f : AlternatingMap R M N ι
                              v : ι → M
                              inst✝ : DecidableEq ι
                              i j : ι
                              hij : Ne i j
                              ⊢ Eq (Function.update v i (v j) i) (Function.update v i (v j) j)
                            -/
  f.map_eq_zero_of_eq _ (by rw [Function.update_self, Function.update_of_ne hij.symm]) hij
                            /-
                              🎉 no goals
                            -/


theorem map_update_update [DecidableEq ι] {i j : ι} (hij : i ≠ j) (m : M) :
    f (Function.update (Function.update v i m) j m) = 0 :=
  f.map_eq_zero_of_eq _
        /-
          R : Type u_1
          inst✝⁵ : Semiring R
          M : Type u_2
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module R M
          N : Type u_3
          inst✝² : AddCommMonoid N
          inst✝¹ : Module R N
          ι : Type u_7
          f : AlternatingMap R M N ι
          v : ι → M
          inst✝ : DecidableEq ι
          i j : ι
          hij : Ne i j
          m : M
          ⊢ Eq (Function.update (Function.update v i m) j m i) (Function.update (Functio …
        -/
    (by rw [Function.update_self, Function.update_of_ne hij, Function.update_self]) hij
        /-
          🎉 no goals
        -/


theorem map_swap_add [DecidableEq ι] {i j : ι} (hij : i ≠ j) :
    f (v ∘ Equiv.swap i j) + f v = 0 := by
  /-
    R : Type u_1
    inst✝⁵ : Semiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    ⊢ Eq (HAdd.hAdd (f (Function.comp v ⇑(Equiv.swap i j))) (f v)) 0
  -/
  rw [Equiv.comp_swap_eq_update]
  /-
    R : Type u_1
    inst✝⁵ : Semiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    ⊢ Eq (HAdd.hAdd (f (Function.update (Function.update v j (v i)) i (v j))) (f v …
  -/
  convert f.map_update_update v hij (v i + v j)
  simp [f.map_update_self _ hij, f.map_update_self _ hij.symm,
    Function.update_comm hij (v i + v j) (v _) v, Function.update_comm hij.symm (v i) (v i) v]


theorem map_add_swap [DecidableEq ι] {i j : ι} (hij : i ≠ j) :
    f v + f (v ∘ Equiv.swap i j) = 0 := by
  /-
    R : Type u_1
    inst✝⁵ : Semiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    ⊢ Eq (HAdd.hAdd (f v) (f (Function.comp v ⇑(Equiv.swap i j)))) 0
  -/
  rw [add_comm]
  /-
    R : Type u_1
    inst✝⁵ : Semiring R
    M : Type u_2
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    N : Type u_3
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R N
    ι : Type u_7
    f : AlternatingMap R M N ι
    v : ι → M
    inst✝ : DecidableEq ι
    i j : ι
    hij : Ne i j
    ⊢ Eq (HAdd.hAdd (f (Function.comp v ⇑(Equiv.swap i j))) (f v)) 0
  -/
  exact f.map_swap_add v hij
  /-
    🎉 no goals
  -/


theorem map_swap [DecidableEq ι] {i j : ι} (hij : i ≠ j) : g (v ∘ Equiv.swap i j) = -g v :=
  eq_neg_of_add_eq_zero_left <| g.map_swap_add v hij


theorem map_perm [DecidableEq ι] [Fintype ι] (v : ι → M) (σ : Equiv.Perm ι) :
    g (v ∘ σ) = Equiv.Perm.sign σ • g v := by
  -- Porting note: `apply` → `induction'`
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N' : Type u_6
    inst✝³ : AddCommGroup N'
    inst✝² : Module R N'
    ι : Type u_7
    g : AlternatingMap R M N' ι
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    v : ι → M
    σ : Equiv.Perm ι
    ⊢ Eq (g (Function.comp v ⇑σ)) (HSMul.hSMul (Equiv.Perm.sign σ) (g v))
  -/
  induction' σ using Equiv.Perm.swap_induction_on' with s x y hxy hI
    /-
      case a
      R : Type u_1
      inst✝⁶ : Semiring R
      M : Type u_2
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      N' : Type u_6
      inst✝³ : AddCommGroup N'
      inst✝² : Module R N'
      ι : Type u_7
      g : AlternatingMap R M N' ι
      inst✝¹ : DecidableEq ι
      inst✝ : Fintype ι
      v : ι → M
      ⊢ Eq (g (Function.comp v ⇑1)) (HSMul.hSMul (Equiv.Perm.sign 1) (g v))
    -/
  · simp
    /-
      🎉 no goals
    -/
  · -- Porting note: `← Function.comp_assoc` & `-Equiv.Perm.sign_swap'` are required.
    simpa [← Function.comp_assoc, g.map_swap (v ∘ s) hxy,
      Equiv.Perm.sign_swap hxy, -Equiv.Perm.sign_swap'] using hI


theorem map_congr_perm [DecidableEq ι] [Fintype ι] (σ : Equiv.Perm ι) :
    g v = Equiv.Perm.sign σ • g (v ∘ σ) := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N' : Type u_6
    inst✝³ : AddCommGroup N'
    inst✝² : Module R N'
    ι : Type u_7
    g : AlternatingMap R M N' ι
    v : ι → M
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    σ : Equiv.Perm ι
    ⊢ Eq (g v) (HSMul.hSMul (Equiv.Perm.sign σ) (g (Function.comp v ⇑σ)))
  -/
  rw [g.map_perm, smul_smul]
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N' : Type u_6
    inst✝³ : AddCommGroup N'
    inst✝² : Module R N'
    ι : Type u_7
    g : AlternatingMap R M N' ι
    v : ι → M
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    σ : Equiv.Perm ι
    ⊢ Eq (g v) (HSMul.hSMul (HMul.hMul (Equiv.Perm.sign σ) (Equiv.Perm.sign σ)) (g …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Transfer the arguments to a map along an equivalence between argument indices.

This is the alternating version of `MultilinearMap.domDomCongr`. -/
@[simps]
def domDomCongr (σ : ι ≃ ι') (f : M [⋀^ι]→ₗ[R] N) : M [⋀^ι']→ₗ[R] N :=
  { f.toMultilinearMap.domDomCongr σ with
    toFun := fun v => f (v ∘ σ)
    map_eq_zero_of_eq' := fun v i j hv hij =>
      f.map_eq_zero_of_eq (v ∘ σ) (i := σ.symm i) (j := σ.symm j)
            /-
              R : Type u_1
              inst✝¹⁴ : Semiring R
              M : Type u_2
              inst✝¹³ : AddCommMonoid M
              inst✝¹² : Module R M
              N : Type u_3
              inst✝¹¹ : AddCommMonoid N
              inst✝¹⁰ : Module R N
              P : Type u_4
              inst✝⁹ : AddCommMonoid P
              inst✝⁸ : Module R P
              M' : Type u_5
              inst✝⁷ : AddCommGroup M'
              inst✝⁶ : Module R M'
              N' : Type u_6
              inst✝⁵ : AddCommGroup N'
              inst✝⁴ : Module R N'
              ι : Type u_7
              ι' : Type u_8
              ι'' : Type u_9
              M₂ : Type u_10
              inst✝³ : AddCommMonoid M₂
              inst✝² : Module R M₂
              M₃ : Type u_11
              inst✝¹ : AddCommMonoid M₃
              inst✝ : Module R M₃
              f✝ f' : AlternatingMap R M N ι
              g g₂ : AlternatingMap R M N' ι
              g' : AlternatingMap R M' N' ι
              v✝ : ι → M
              v' : ι → M'
              σ : Equiv ι ι'
              f : AlternatingMap R M N ι
              v : ι' → M
              i j : ι'
              hv : Eq (v i) (v j)
              hij : Ne i j
              ⊢ Eq (Function.comp v (⇑σ) (σ.symm i)) (Function.comp v (⇑σ) (σ.symm j))
            -/
        (by simpa using hv) (σ.symm.injective.ne hij) }
            /-
              🎉 no goals
            -/


@[simp]
theorem domDomCongr_refl (f : M [⋀^ι]→ₗ[R] N) : f.domDomCongr (Equiv.refl ι) = f := rfl


theorem domDomCongr_trans (σ₁ : ι ≃ ι') (σ₂ : ι' ≃ ι'') (f : M [⋀^ι]→ₗ[R] N) :
    f.domDomCongr (σ₁.trans σ₂) = (f.domDomCongr σ₁).domDomCongr σ₂ :=
  rfl


@[simp]
theorem domDomCongr_zero (σ : ι ≃ ι') : (0 : M [⋀^ι]→ₗ[R] N).domDomCongr σ = 0 :=
  rfl


@[simp]
theorem domDomCongr_add (σ : ι ≃ ι') (f g : M [⋀^ι]→ₗ[R] N) :
    (f + g).domDomCongr σ = f.domDomCongr σ + g.domDomCongr σ :=
  rfl


@[simp]
theorem domDomCongr_smul {S : Type*} [Monoid S] [DistribMulAction S N] [SMulCommClass R S N]
    (σ : ι ≃ ι') (c : S) (f : M [⋀^ι]→ₗ[R] N) :
    (c • f).domDomCongr σ = c • f.domDomCongr σ :=
  rfl


/-- `AlternatingMap.domDomCongr` as an equivalence.

This is declared separately because it does not work with dot notation. -/
@[simps apply symm_apply]
def domDomCongrEquiv (σ : ι ≃ ι') : M [⋀^ι]→ₗ[R] N ≃+ M [⋀^ι']→ₗ[R] N where
  toFun := domDomCongr σ
  invFun := domDomCongr σ.symm
  left_inv f := by
    /-
      R : Type u_1
      inst✝¹⁴ : Semiring R
      M : Type u_2
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : Module R M
      N : Type u_3
      inst✝¹¹ : AddCommMonoid N
      inst✝¹⁰ : Module R N
      P : Type u_4
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      M' : Type u_5
      inst✝⁷ : AddCommGroup M'
      inst✝⁶ : Module R M'
      N' : Type u_6
      inst✝⁵ : AddCommGroup N'
      inst✝⁴ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      M₂ : Type u_10
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      M₃ : Type u_11
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      f✝ f' : AlternatingMap R M N ι
      g g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      σ : Equiv ι ι'
      f : AlternatingMap R M N ι
      ⊢ Eq (AlternatingMap.domDomCongr σ.symm (AlternatingMap.domDomCongr σ f)) f
    -/
    ext
    /-
      case H
      R : Type u_1
      inst✝¹⁴ : Semiring R
      M : Type u_2
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : Module R M
      N : Type u_3
      inst✝¹¹ : AddCommMonoid N
      inst✝¹⁰ : Module R N
      P : Type u_4
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      M' : Type u_5
      inst✝⁷ : AddCommGroup M'
      inst✝⁶ : Module R M'
      N' : Type u_6
      inst✝⁵ : AddCommGroup N'
      inst✝⁴ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      M₂ : Type u_10
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      M₃ : Type u_11
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      f✝ f' : AlternatingMap R M N ι
      g g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      σ : Equiv ι ι'
      f : AlternatingMap R M N ι
      x✝ : ι → M
      ⊢ Eq ((AlternatingMap.domDomCongr σ.symm (AlternatingMap.domDomCongr σ f)) x✝) …
    -/
    simp [Function.comp_def]
    /-
      🎉 no goals
    -/
  right_inv m := by
    /-
      R : Type u_1
      inst✝¹⁴ : Semiring R
      M : Type u_2
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : Module R M
      N : Type u_3
      inst✝¹¹ : AddCommMonoid N
      inst✝¹⁰ : Module R N
      P : Type u_4
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      M' : Type u_5
      inst✝⁷ : AddCommGroup M'
      inst✝⁶ : Module R M'
      N' : Type u_6
      inst✝⁵ : AddCommGroup N'
      inst✝⁴ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      M₂ : Type u_10
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      M₃ : Type u_11
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      f f' : AlternatingMap R M N ι
      g g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      σ : Equiv ι ι'
      m : AlternatingMap R M N ι'
      ⊢ Eq (AlternatingMap.domDomCongr σ (AlternatingMap.domDomCongr σ.symm m)) m
    -/
    ext
    /-
      case H
      R : Type u_1
      inst✝¹⁴ : Semiring R
      M : Type u_2
      inst✝¹³ : AddCommMonoid M
      inst✝¹² : Module R M
      N : Type u_3
      inst✝¹¹ : AddCommMonoid N
      inst✝¹⁰ : Module R N
      P : Type u_4
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      M' : Type u_5
      inst✝⁷ : AddCommGroup M'
      inst✝⁶ : Module R M'
      N' : Type u_6
      inst✝⁵ : AddCommGroup N'
      inst✝⁴ : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      M₂ : Type u_10
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      M₃ : Type u_11
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      f f' : AlternatingMap R M N ι
      g g₂ : AlternatingMap R M N' ι
      g' : AlternatingMap R M' N' ι
      v : ι → M
      v' : ι → M'
      σ : Equiv ι ι'
      m : AlternatingMap R M N ι'
      x✝ : ι' → M
      ⊢ Eq ((AlternatingMap.domDomCongr σ (AlternatingMap.domDomCongr σ.symm m)) x✝) …
    -/
    simp [Function.comp_def]
    /-
      🎉 no goals
    -/
  map_add' := domDomCongr_add σ


/-- `AlternatingMap.domDomCongr` as a linear equivalence. -/
@[simps apply symm_apply]
def domDomCongrₗ (σ : ι ≃ ι') : M [⋀^ι]→ₗ[R] N ≃ₗ[S] M [⋀^ι']→ₗ[R] N where
  toFun := domDomCongr σ
  invFun := domDomCongr σ.symm
                   /-
                     R : Type u_1
                     inst✝¹⁷ : Semiring R
                     M : Type u_2
                     inst✝¹⁶ : AddCommMonoid M
                     inst✝¹⁵ : Module R M
                     N : Type u_3
                     inst✝¹⁴ : AddCommMonoid N
                     inst✝¹³ : Module R N
                     P : Type u_4
                     inst✝¹² : AddCommMonoid P
                     inst✝¹¹ : Module R P
                     M' : Type u_5
                     inst✝¹⁰ : AddCommGroup M'
                     inst✝⁹ : Module R M'
                     N' : Type u_6
                     inst✝⁸ : AddCommGroup N'
                     inst✝⁷ : Module R N'
                     ι : Type u_7
                     ι' : Type u_8
                     ι'' : Type u_9
                     M₂ : Type u_10
                     inst✝⁶ : AddCommMonoid M₂
                     inst✝⁵ : Module R M₂
                     M₃ : Type u_11
                     inst✝⁴ : AddCommMonoid M₃
                     inst✝³ : Module R M₃
                     f✝ f' : AlternatingMap R M N ι
                     g g₂ : AlternatingMap R M N' ι
                     g' : AlternatingMap R M' N' ι
                     v : ι → M
                     v' : ι → M'
                     S : Type u_12
                     inst✝² : Semiring S
                     inst✝¹ : Module S N
                     inst✝ : SMulCommClass R S N
                     σ : Equiv ι ι'
                     f : AlternatingMap R M N ι
                     ⊢ Eq (AlternatingMap.domDomCongr σ.symm ({ toFun := AlternatingMap.domDomCongr …
                   -/
  left_inv f := by ext; simp [Function.comp_def]
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u_1
                      inst✝¹⁷ : Semiring R
                      M : Type u_2
                      inst✝¹⁶ : AddCommMonoid M
                      inst✝¹⁵ : Module R M
                      N : Type u_3
                      inst✝¹⁴ : AddCommMonoid N
                      inst✝¹³ : Module R N
                      P : Type u_4
                      inst✝¹² : AddCommMonoid P
                      inst✝¹¹ : Module R P
                      M' : Type u_5
                      inst✝¹⁰ : AddCommGroup M'
                      inst✝⁹ : Module R M'
                      N' : Type u_6
                      inst✝⁸ : AddCommGroup N'
                      inst✝⁷ : Module R N'
                      ι : Type u_7
                      ι' : Type u_8
                      ι'' : Type u_9
                      M₂ : Type u_10
                      inst✝⁶ : AddCommMonoid M₂
                      inst✝⁵ : Module R M₂
                      M₃ : Type u_11
                      inst✝⁴ : AddCommMonoid M₃
                      inst✝³ : Module R M₃
                      f f' : AlternatingMap R M N ι
                      g g₂ : AlternatingMap R M N' ι
                      g' : AlternatingMap R M' N' ι
                      v : ι → M
                      v' : ι → M'
                      S : Type u_12
                      inst✝² : Semiring S
                      inst✝¹ : Module S N
                      inst✝ : SMulCommClass R S N
                      σ : Equiv ι ι'
                      m : AlternatingMap R M N ι'
                      ⊢ Eq ({ toFun := AlternatingMap.domDomCongr σ, map_add' := ⋯, map_smul' := ⋯ } …
                    -/
  right_inv m := by ext; simp [Function.comp_def]
                         /-
                           🎉 no goals
                         -/
  map_add' := domDomCongr_add σ
  map_smul' := domDomCongr_smul σ


@[simp]
theorem domDomCongrₗ_refl :
    (domDomCongrₗ S (Equiv.refl ι) : M [⋀^ι]→ₗ[R] N ≃ₗ[S] M [⋀^ι]→ₗ[R] N) =
      LinearEquiv.refl _ _ :=
  rfl


@[simp]
theorem domDomCongrₗ_toAddEquiv (σ : ι ≃ ι') :
    (↑(domDomCongrₗ S σ : M [⋀^ι]→ₗ[R] N ≃ₗ[S] _) : M [⋀^ι]→ₗ[R] N ≃+ _) =
      domDomCongrEquiv σ :=
  rfl


/-- The results of applying `domDomCongr` to two maps are equal if and only if those maps are. -/
@[simp]
theorem domDomCongr_eq_iff (σ : ι ≃ ι') (f g : M [⋀^ι]→ₗ[R] N) :
    f.domDomCongr σ = g.domDomCongr σ ↔ f = g :=
  (domDomCongrEquiv σ : _ ≃+ M [⋀^ι']→ₗ[R] N).apply_eq_iff_eq


@[simp]
theorem domDomCongr_eq_zero_iff (σ : ι ≃ ι') (f : M [⋀^ι]→ₗ[R] N) :
    f.domDomCongr σ = 0 ↔ f = 0 :=
  (domDomCongrEquiv σ : M [⋀^ι]→ₗ[R] N ≃+ M [⋀^ι']→ₗ[R] N).map_eq_zero_iff


theorem domDomCongr_perm [Fintype ι] [DecidableEq ι] (σ : Equiv.Perm ι) :
    g.domDomCongr σ = Equiv.Perm.sign σ • g :=
  AlternatingMap.ext fun v => g.map_perm v σ


@[norm_cast]
theorem coe_domDomCongr (σ : ι ≃ ι') :
    ↑(f.domDomCongr σ) = (f : MultilinearMap R (fun _ : ι => M) N).domDomCongr σ :=
  MultilinearMap.ext fun _ => rfl


/-- If the arguments are linearly dependent then the result is `0`. -/
theorem map_linearDependent {K : Type*} [Ring K] {M : Type*} [AddCommGroup M] [Module K M]
    {N : Type*} [AddCommGroup N] [Module K N] [NoZeroSMulDivisors K N] (f : M [⋀^ι]→ₗ[K] N)
    (v : ι → M) (h : ¬LinearIndependent K v) : f v = 0 := by
  /-
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h : Not (LinearIndependent K v)
    ⊢ Eq (f v) 0
  -/
  obtain ⟨s, g, h, i, hi, hz⟩ := not_linearIndependent_iff.mp h
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h✝ : Not (LinearIndependent K v)
    s : Finset ι
    g : ι → K
    h : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
    i : ι
    hi : Membership.mem s i
    hz : Ne (g i) 0
    ⊢ Eq (f v) 0
  -/
  letI := Classical.decEq ι
  suffices f (update v i (g i • v i)) = 0 by
    rw [f.map_update_smul, Function.update_eq_self, smul_eq_zero] at this
    exact Or.resolve_left this hz
  -- Porting note: Was `conv at h in .. => ..`.
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h✝ : Not (LinearIndependent K v)
    s : Finset ι
    g : ι → K
    h : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
    i : ι
    hi : Membership.mem s i
    hz : Ne (g i) 0
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq (f (Function.update v i (HSMul.hSMul (g i) (v i)))) 0
  -/
  rw [← (funext fun x => ite_self (c := i = x) (d := Classical.decEq ι i x) (g x • v x))] at h
  rw [Finset.sum_ite, Finset.filter_eq, Finset.filter_ne, if_pos hi, Finset.sum_singleton,
    add_eq_zero_iff_eq_neg] at h
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h✝ : Not (LinearIndependent K v)
    s : Finset ι
    g : ι → K
    i : ι
    hi : Membership.mem s i
    hz : Ne (g i) 0
    this : DecidableEq ι := Classical.decEq ι
    h : Eq (HSMul.hSMul (g i) (v i)) (Neg.neg ((s.erase i).sum fun x => HSMul.hSMu …
    ⊢ Eq (f (Function.update v i (HSMul.hSMul (g i) (v i)))) 0
  -/
  rw [h, f.map_update_neg, f.map_update_sum, neg_eq_zero]; apply Finset.sum_eq_zero
  /-
    case intro.intro.intro.intro.intro.h
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h✝ : Not (LinearIndependent K v)
    s : Finset ι
    g : ι → K
    i : ι
    hi : Membership.mem s i
    hz : Ne (g i) 0
    this : DecidableEq ι := Classical.decEq ι
    h : Eq (HSMul.hSMul (g i) (v i)) (Neg.neg ((s.erase i).sum fun x => HSMul.hSMu …
    ⊢ ∀ (x : ι), Membership.mem (s.erase i) x → Eq (f (Function.update v i (HSMul. …
  -/
  intro j hj
  /-
    case intro.intro.intro.intro.intro.h
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h✝ : Not (LinearIndependent K v)
    s : Finset ι
    g : ι → K
    i : ι
    hi : Membership.mem s i
    hz : Ne (g i) 0
    this : DecidableEq ι := Classical.decEq ι
    h : Eq (HSMul.hSMul (g i) (v i)) (Neg.neg ((s.erase i).sum fun x => HSMul.hSMu …
    j : ι
    hj : Membership.mem (s.erase i) j
    ⊢ Eq (f (Function.update v i (HSMul.hSMul (g j) (v j)))) 0
  -/
  obtain ⟨hij, _⟩ := Finset.mem_erase.mp hj
  /-
    case intro.intro.intro.intro.intro.h.intro
    ι : Type u_7
    K : Type u_12
    inst✝⁵ : Ring K
    M : Type u_13
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module K M
    N : Type u_14
    inst✝² : AddCommGroup N
    inst✝¹ : Module K N
    inst✝ : NoZeroSMulDivisors K N
    f : AlternatingMap K M N ι
    v : ι → M
    h✝ : Not (LinearIndependent K v)
    s : Finset ι
    g : ι → K
    i : ι
    hi : Membership.mem s i
    hz : Ne (g i) 0
    this : DecidableEq ι := Classical.decEq ι
    h : Eq (HSMul.hSMul (g i) (v i)) (Neg.neg ((s.erase i).sum fun x => HSMul.hSMu …
    j : ι
    hj : Membership.mem (s.erase i) j
    hij : Ne j i
    right✝ : Membership.mem s j
    ⊢ Eq (f (Function.update v i (HSMul.hSMul (g j) (v j)))) 0
  -/
  rw [f.map_update_smul, f.map_update_self _ hij.symm, smul_zero]
  /-
    🎉 no goals
  -/


/-- A version of `MultilinearMap.cons_add` for `AlternatingMap`. -/
theorem map_vecCons_add {n : ℕ} (f : M [⋀^Fin n.succ]→ₗ[R] N) (m : Fin n → M) (x y : M) :
    f (Matrix.vecCons (x + y) m) = f (Matrix.vecCons x m) + f (Matrix.vecCons y m) :=
  f.toMultilinearMap.cons_add _ _ _


/-- A version of `MultilinearMap.cons_smul` for `AlternatingMap`. -/
theorem map_vecCons_smul {n : ℕ} (f : M [⋀^Fin n.succ]→ₗ[R] N) (m : Fin n → M) (c : R)
    (x : M) : f (Matrix.vecCons (c • x) m) = c • f (Matrix.vecCons x m) :=
  f.toMultilinearMap.cons_smul _ _ _


private theorem alternization_map_eq_zero_of_eq_aux (m : MultilinearMap R (fun _ : ι => M) N')
    (v : ι → M) (i j : ι) (i_ne_j : i ≠ j) (hv : v i = v j) :
    (∑ σ : Perm ι, Equiv.Perm.sign σ • m.domDomCongr σ) v = 0 := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N' : Type u_6
    inst✝³ : AddCommGroup N'
    inst✝² : Module R N'
    ι : Type u_7
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    m : MultilinearMap R (fun x => M) N'
    v : ι → M
    i j : ι
    i_ne_j : Ne i j
    hv : Eq (v i) (v j)
    ⊢ Eq ((Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign σ) (MultilinearMa …
  -/
  rw [sum_apply]
  exact
    Finset.sum_involution (fun σ _ => swap i j * σ)
      -- Porting note: `-Equiv.Perm.sign_swap'` is required.
      (fun σ _ => by simp [Perm.sign_swap i_ne_j, apply_swap_eq_self hv, -Equiv.Perm.sign_swap'])
      (fun σ _ _ => (not_congr swap_mul_eq_iff).mpr i_ne_j) (fun σ _ => Finset.mem_univ _)
      fun σ _ => swap_mul_involutive i j σ


/-- Produce an `AlternatingMap` out of a `MultilinearMap`, by summing over all argument
permutations. -/
def alternatization : MultilinearMap R (fun _ : ι => M) N' →+ M [⋀^ι]→ₗ[R] N' where
  toFun m :=
    { ∑ σ : Perm ι, Equiv.Perm.sign σ • m.domDomCongr σ with
      toFun := ⇑(∑ σ : Perm ι, Equiv.Perm.sign σ • m.domDomCongr σ)
      map_eq_zero_of_eq' := fun v i j hvij hij =>
        alternization_map_eq_zero_of_eq_aux m v i j hij hvij }
  map_add' a b := by
    /-
      R : Type u_1
      inst✝¹² : Semiring R
      M : Type u_2
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : Module R M
      N : Type u_3
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : Module R N
      P : Type u_4
      inst✝⁷ : AddCommMonoid P
      inst✝⁶ : Module R P
      M' : Type u_5
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module R M'
      N' : Type u_6
      inst✝³ : AddCommGroup N'
      inst✝² : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      a b : MultilinearMap R (fun x => M) N'
      ⊢ Eq
          ({
                toFun := fun m =>
                  let __src := Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign …
                  { toFun := ⇑(Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign …
                map_zero' := ⋯ }.toFun
            (HAdd.hAdd a b))
          (HAdd.hAdd
            ({
                  toFun := fun m =>
                    let __src := Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.si …
                    { toFun := ⇑(Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.si …
                  map_zero' := ⋯ }.toFun
              a)
            ({
                  toFun := fun m =>
                    let __src := Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.si …
                    { toFun := ⇑(Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.si …
                  map_zero' := ⋯ }.toFun
              b))
    -/
    ext
    simp only [mk_coe, AlternatingMap.coe_mk, sum_apply, smul_apply, domDomCongr_apply, add_apply,
      smul_add, Finset.sum_add_distrib, AlternatingMap.add_apply]
    /-
      R : Type u_1
      inst✝¹² : Semiring R
      M : Type u_2
      inst✝¹¹ : AddCommMonoid M
      inst✝¹⁰ : Module R M
      N : Type u_3
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : Module R N
      P : Type u_4
      inst✝⁷ : AddCommMonoid P
      inst✝⁶ : Module R P
      M' : Type u_5
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module R M'
      N' : Type u_6
      inst✝³ : AddCommGroup N'
      inst✝² : Module R N'
      ι : Type u_7
      ι' : Type u_8
      ι'' : Type u_9
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      ⊢ Eq
          ((fun m =>
              let __src := Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign σ)  …
              { toFun := ⇑(Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign σ)  …
            0)
          0
    -/
  map_zero' := by
    ext
    simp only [mk_coe, AlternatingMap.coe_mk, sum_apply, smul_apply, domDomCongr_apply,
      zero_apply, smul_zero, Finset.sum_const_zero, AlternatingMap.zero_apply]


theorem alternatization_def (m : MultilinearMap R (fun _ : ι => M) N') :
    ⇑(alternatization m) = (∑ σ : Perm ι, Equiv.Perm.sign σ • m.domDomCongr σ : _) :=
  rfl


theorem alternatization_coe (m : MultilinearMap R (fun _ : ι => M) N') :
    ↑(alternatization m) = (∑ σ : Perm ι, Equiv.Perm.sign σ • m.domDomCongr σ : _) :=
  coe_injective rfl


theorem alternatization_apply (m : MultilinearMap R (fun _ : ι => M) N') (v : ι → M) :
    alternatization m v = ∑ σ : Perm ι, Equiv.Perm.sign σ • m.domDomCongr σ v := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N' : Type u_6
    inst✝³ : AddCommGroup N'
    inst✝² : Module R N'
    ι : Type u_7
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    m : MultilinearMap R (fun x => M) N'
    v : ι → M
    ⊢ Eq ((MultilinearMap.alternatization m) v) (Finset.univ.sum fun σ => HSMul.hS …
  -/
  simp only [alternatization_def, smul_apply, sum_apply]
  /-
    🎉 no goals
  -/


/-- Alternatizing a multilinear map that is already alternating results in a scale factor of `n!`,
where `n` is the number of inputs. -/
theorem coe_alternatization [DecidableEq ι] [Fintype ι] (a : M [⋀^ι]→ₗ[R] N') :
    MultilinearMap.alternatization (a : MultilinearMap R (fun _ => M) N')
    = Nat.factorial (Fintype.card ι) • a := by
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    M : Type u_2
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    N' : Type u_6
    inst✝³ : AddCommGroup N'
    inst✝² : Module R N'
    ι : Type u_7
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    a : AlternatingMap R M N' ι
    ⊢ Eq (MultilinearMap.alternatization ↑a) (HSMul.hSMul (Fintype.card ι).factori …
  -/
  apply AlternatingMap.coe_injective
  simp_rw [MultilinearMap.alternatization_def, ← coe_domDomCongr, domDomCongr_perm, coe_smul,
    smul_smul, Int.units_mul_self, one_smul, Finset.sum_const, Finset.card_univ, Fintype.card_perm,
    ← coe_multilinearMap, coe_smul]


/-- Composition with a linear map before and after alternatization are equivalent. -/
theorem compMultilinearMap_alternatization (g : N' →ₗ[R] N'₂)
    (f : MultilinearMap R (fun _ : ι => M) N') :
    MultilinearMap.alternatization (g.compMultilinearMap f)
      = g.compAlternatingMap (MultilinearMap.alternatization f) := by
  /-
    R : Type u_1
    inst✝⁸ : Semiring R
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N' : Type u_6
    inst✝⁵ : AddCommGroup N'
    inst✝⁴ : Module R N'
    ι : Type u_7
    N'₂ : Type u_10
    inst✝³ : AddCommGroup N'₂
    inst✝² : Module R N'₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    g : LinearMap (RingHom.id R) N' N'₂
    f : MultilinearMap R (fun x => M) N'
    ⊢ Eq (MultilinearMap.alternatization (g.compMultilinearMap f)) (g.compAlternat …
  -/
  ext
  /-
    case H
    R : Type u_1
    inst✝⁸ : Semiring R
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N' : Type u_6
    inst✝⁵ : AddCommGroup N'
    inst✝⁴ : Module R N'
    ι : Type u_7
    N'₂ : Type u_10
    inst✝³ : AddCommGroup N'₂
    inst✝² : Module R N'₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    g : LinearMap (RingHom.id R) N' N'₂
    f : MultilinearMap R (fun x => M) N'
    x✝ : ι → M
    ⊢ Eq ((MultilinearMap.alternatization (g.compMultilinearMap f)) x✝) ((g.compAl …
  -/
  simp [MultilinearMap.alternatization_def]
  /-
    🎉 no goals
  -/


/-- Two alternating maps indexed by a `Fintype` are equal if they are equal when all arguments
are distinct basis vectors. -/
theorem Basis.ext_alternating {f g : N₁ [⋀^ι]→ₗ[R'] N₂} (e : Basis ι₁ R' N₁)
    (h : ∀ v : ι → ι₁, Function.Injective v → (f fun i => e (v i)) = g fun i => e (v i)) :
    f = g := by
  classical
    refine AlternatingMap.coe_multilinearMap_injective (Basis.ext_multilinear e fun v => ?_)
    by_cases hi : Function.Injective v
    · exact h v hi
    · have : ¬Function.Injective fun i => e (v i) := hi.imp Function.Injective.of_comp
      rw [coe_multilinearMap, coe_multilinearMap, f.map_eq_zero_of_not_injective _ this,
        g.map_eq_zero_of_not_injective _ this]


/-- Given an alternating map `f` in `n+1` variables, split the first variable to obtain
a linear map into alternating maps in `n` variables, given by `x ↦ (m ↦ f (Matrix.vecCons x m))`.
It can be thought of as a map $Hom(\bigwedge^{n+1} M, N) \to Hom(M, Hom(\bigwedge^n M, N))$.

This is `MultilinearMap.curryLeft` for `AlternatingMap`. See also
`AlternatingMap.curryLeftLinearMap`. -/
@[simps]
def curryLeft {n : ℕ} (f : M'' [⋀^Fin n.succ]→ₗ[R'] N'') :
    M'' →ₗ[R'] M'' [⋀^Fin n]→ₗ[R'] N'' where
  toFun m :=
    { f.toMultilinearMap.curryLeft m with
      toFun := fun v => f (Matrix.vecCons m v)
      map_eq_zero_of_eq' := fun v i j hv hij =>
        f.map_eq_zero_of_eq _ (by
          /-
            R : Type u_1
            inst✝¹⁹ : Semiring R
            M : Type u_2
            inst✝¹⁸ : AddCommMonoid M
            inst✝¹⁷ : Module R M
            N : Type u_3
            inst✝¹⁶ : AddCommMonoid N
            inst✝¹⁵ : Module R N
            P : Type u_4
            inst✝¹⁴ : AddCommMonoid P
            inst✝¹³ : Module R P
            M' : Type u_5
            inst✝¹² : AddCommGroup M'
            inst✝¹¹ : Module R M'
            N' : Type u_6
            inst✝¹⁰ : AddCommGroup N'
            inst✝⁹ : Module R N'
            ι : Type u_7
            ι' : Type u_8
            ι'' : Type u_9
            R' : Type u_10
            M'' : Type u_11
            M₂'' : Type u_12
            N'' : Type u_13
            N₂'' : Type u_14
            inst✝⁸ : CommSemiring R'
            inst✝⁷ : AddCommMonoid M''
            inst✝⁶ : AddCommMonoid M₂''
            inst✝⁵ : AddCommMonoid N''
            inst✝⁴ : AddCommMonoid N₂''
            inst✝³ : Module R' M''
            inst✝² : Module R' M₂''
            inst✝¹ : Module R' N''
            inst✝ : Module R' N₂''
            n : Nat
            f : AlternatingMap R' M'' N'' (Fin n.succ)
            m : M''
            v : Fin n → M''
            i j : Fin n
            hv : Eq (v i) (v j)
            hij : Ne i j
            ⊢ Eq (Matrix.vecCons m v i.succ) (Matrix.vecCons m v j.succ)
          -/
          rwa [Matrix.cons_val_succ, Matrix.cons_val_succ]) ((Fin.succ_injective _).ne hij) }
          /-
            🎉 no goals
          -/
  map_add' _ _ := ext fun _ => f.map_vecCons_add _ _ _
  map_smul' _ _ := ext fun _ => f.map_vecCons_smul _ _ _


@[simp]
theorem curryLeft_zero {n : ℕ} : curryLeft (0 : M'' [⋀^Fin n.succ]→ₗ[R'] N'') = 0 :=
  rfl


@[simp]
theorem curryLeft_add {n : ℕ} (f g : M'' [⋀^Fin n.succ]→ₗ[R'] N'') :
    curryLeft (f + g) = curryLeft f + curryLeft g :=
  rfl


@[simp]
theorem curryLeft_smul {n : ℕ} (r : R') (f : M'' [⋀^Fin n.succ]→ₗ[R'] N'') :
    curryLeft (r • f) = r • curryLeft f :=
  rfl


/-- `AlternatingMap.curryLeft` as a `LinearMap`. This is a separate definition as dot notation
does not work for this version. -/
@[simps]
def curryLeftLinearMap {n : ℕ} :
    (M'' [⋀^Fin n.succ]→ₗ[R'] N'') →ₗ[R'] M'' →ₗ[R'] M'' [⋀^Fin n]→ₗ[R'] N'' where
  toFun f := f.curryLeft
  map_add' := curryLeft_add
  map_smul' := curryLeft_smul


/-- Currying with the same element twice gives the zero map. -/
@[simp]
theorem curryLeft_same {n : ℕ} (f : M'' [⋀^Fin n.succ.succ]→ₗ[R'] N'') (m : M'') :
    (f.curryLeft m).curryLeft m = 0 :=
                                         /-
                                           R' : Type u_10
                                           M'' : Type u_11
                                           N'' : Type u_13
                                           inst✝⁴ : CommSemiring R'
                                           inst✝³ : AddCommMonoid M''
                                           inst✝² : AddCommMonoid N''
                                           inst✝¹ : Module R' M''
                                           inst✝ : Module R' N''
                                           n : Nat
                                           f : AlternatingMap R' M'' N'' (Fin n.succ.succ)
                                           m : M''
                                           x✝ : Fin n → M''
                                           ⊢ Eq (Matrix.vecCons m (Matrix.vecCons m x✝) 0) (Matrix.vecCons m (Matrix.vecC …
                                         -/
  ext fun _ => f.map_eq_zero_of_eq _ (by simp) Fin.zero_ne_one
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem curryLeft_compAlternatingMap {n : ℕ} (g : N'' →ₗ[R'] N₂'')
    (f : M'' [⋀^Fin n.succ]→ₗ[R'] N'') (m : M'') :
    (g.compAlternatingMap f).curryLeft m = g.compAlternatingMap (f.curryLeft m) :=
  rfl


@[simp]
theorem curryLeft_compLinearMap {n : ℕ} (g : M₂'' →ₗ[R'] M'')
    (f : M'' [⋀^Fin n.succ]→ₗ[R'] N'') (m : M₂'') :
    (f.compLinearMap g).curryLeft m = (f.curryLeft (g m)).compLinearMap g :=
  ext fun v => congr_arg f <| funext <| by
    /-
      R' : Type u_10
      M'' : Type u_11
      M₂'' : Type u_12
      N'' : Type u_13
      inst✝⁶ : CommSemiring R'
      inst✝⁵ : AddCommMonoid M''
      inst✝⁴ : AddCommMonoid M₂''
      inst✝³ : AddCommMonoid N''
      inst✝² : Module R' M''
      inst✝¹ : Module R' M₂''
      inst✝ : Module R' N''
      n : Nat
      g : LinearMap (RingHom.id R') M₂'' M''
      f : AlternatingMap R' M'' N'' (Fin n.succ)
      m : M₂''
      v : Fin n → M₂''
      ⊢ ∀ (x : Fin n.succ), Eq (((fun x => g) x) (Matrix.vecCons m v x)) (Matrix.vec …
    -/
    refine Fin.cases ?_ ?_
      /-
        case refine_1
        R' : Type u_10
        M'' : Type u_11
        M₂'' : Type u_12
        N'' : Type u_13
        inst✝⁶ : CommSemiring R'
        inst✝⁵ : AddCommMonoid M''
        inst✝⁴ : AddCommMonoid M₂''
        inst✝³ : AddCommMonoid N''
        inst✝² : Module R' M''
        inst✝¹ : Module R' M₂''
        inst✝ : Module R' N''
        n : Nat
        g : LinearMap (RingHom.id R') M₂'' M''
        f : AlternatingMap R' M'' N'' (Fin n.succ)
        m : M₂''
        v : Fin n → M₂''
        ⊢ Eq (((fun x => g) 0) (Matrix.vecCons m v 0)) (Matrix.vecCons (g m) (fun i => …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R' : Type u_10
        M'' : Type u_11
        M₂'' : Type u_12
        N'' : Type u_13
        inst✝⁶ : CommSemiring R'
        inst✝⁵ : AddCommMonoid M''
        inst✝⁴ : AddCommMonoid M₂''
        inst✝³ : AddCommMonoid N''
        inst✝² : Module R' M''
        inst✝¹ : Module R' M₂''
        inst✝ : Module R' N''
        n : Nat
        g : LinearMap (RingHom.id R') M₂'' M''
        f : AlternatingMap R' M'' N'' (Fin n.succ)
        m : M₂''
        v : Fin n → M₂''
        ⊢ ∀ (i : Fin n), Eq (((fun x => g) i.succ) (Matrix.vecCons m v i.succ)) (Matri …
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- The space of constant maps is equivalent to the space of maps that are alternating with respect
to an empty family. -/
@[simps]
def constLinearEquivOfIsEmpty [IsEmpty ι] : N'' ≃ₗ[R'] (M'' [⋀^ι]→ₗ[R'] N'') where
  toFun := AlternatingMap.constOfIsEmpty R' M'' ι
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun f := f 0
  left_inv _ := rfl
  right_inv f := ext fun _ => AlternatingMap.congr_arg f <| Subsingleton.elim _ _


