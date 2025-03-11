/-- Create a bilinear map from a function that is semilinear in each component.
See `mk₂'` and `mk₂` for the linear case. -/
def mk₂'ₛₗ (f : M → N → P) (H1 : ∀ m₁ m₂ n, f (m₁ + m₂) n = f m₁ n + f m₂ n)
    (H2 : ∀ (c : R) (m n), f (c • m) n = ρ₁₂ c • f m n)
    (H3 : ∀ m n₁ n₂, f m (n₁ + n₂) = f m n₁ + f m n₂)
    (H4 : ∀ (c : S) (m n), f m (c • n) = σ₁₂ c • f m n) : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P where
  toFun m :=
    { toFun := f m
      map_add' := H3 m
      map_smul' := fun c => H4 c m }
  map_add' m₁ m₂ := LinearMap.ext <| H1 m₁ m₂
  map_smul' c m := LinearMap.ext <| H2 c m


@[simp]
theorem mk₂'ₛₗ_apply (f : M → N → P) {H1 H2 H3 H4} (m : M) (n : N) :
    (mk₂'ₛₗ ρ₁₂ σ₁₂ f H1 H2 H3 H4 : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) m n = f m n := rfl


/-- Create a bilinear map from a function that is linear in each component.
See `mk₂` for the special case where both arguments come from modules over the same ring. -/
def mk₂' (f : M → N → Pₗ) (H1 : ∀ m₁ m₂ n, f (m₁ + m₂) n = f m₁ n + f m₂ n)
    (H2 : ∀ (c : R) (m n), f (c • m) n = c • f m n)
    (H3 : ∀ m n₁ n₂, f m (n₁ + n₂) = f m n₁ + f m n₂)
    (H4 : ∀ (c : S) (m n), f m (c • n) = c • f m n) : M →ₗ[R] N →ₗ[S] Pₗ :=
  mk₂'ₛₗ (RingHom.id R) (RingHom.id S) f H1 H2 H3 H4


@[simp]
theorem mk₂'_apply (f : M → N → Pₗ) {H1 H2 H3 H4} (m : M) (n : N) :
    (mk₂' R S f H1 H2 H3 H4 : M →ₗ[R] N →ₗ[S] Pₗ) m n = f m n := rfl


theorem ext₂ {f g : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P} (H : ∀ m n, f m n = g m n) : f = g :=
  LinearMap.ext fun m => LinearMap.ext fun n => H m n


theorem congr_fun₂ {f g : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P} (h : f = g) (x y) : f x y = g x y :=
  LinearMap.congr_fun (LinearMap.congr_fun h x) y


theorem ext_iff₂ {f g : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P} : f = g ↔ ∀ m n, f m n = g m n :=
  ⟨congr_fun₂, ext₂⟩


/-- Given a linear map from `M` to linear maps from `N` to `P`, i.e., a bilinear map from `M × N` to
`P`, change the order of variables and get a linear map from `N` to linear maps from `M` to `P`. -/
def flip (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) : N →ₛₗ[σ₁₂] M →ₛₗ[ρ₁₂] P :=
  mk₂'ₛₗ σ₁₂ ρ₁₂ (fun n m => f m n) (fun _ _ m => (f m).map_add _ _)
    (fun _ _  m  => (f m).map_smulₛₗ _ _)
                       /-
                         R : Type u_1
                         inst✝²⁹ : Semiring R
                         S : Type u_2
                         inst✝²⁸ : Semiring S
                         R₂ : Type u_3
                         inst✝²⁷ : Semiring R₂
                         S₂ : Type u_4
                         inst✝²⁶ : Semiring S₂
                         M : Type u_5
                         N : Type u_6
                         P : Type u_7
                         M₂ : Type u_8
                         N₂ : Type u_9
                         P₂ : Type u_10
                         Pₗ : Type u_11
                         M' : Type u_12
                         P' : Type u_13
                         inst✝²⁵ : AddCommMonoid M
                         inst✝²⁴ : AddCommMonoid N
                         inst✝²³ : AddCommMonoid P
                         inst✝²² : AddCommMonoid M₂
                         inst✝²¹ : AddCommMonoid N₂
                         inst✝²⁰ : AddCommMonoid P₂
                         inst✝¹⁹ : AddCommMonoid Pₗ
                         inst✝¹⁸ : AddCommGroup M'
                         inst✝¹⁷ : AddCommGroup P'
                         inst✝¹⁶ : Module R M
                         inst✝¹⁵ : Module S N
                         inst✝¹⁴ : Module R₂ P
                         inst✝¹³ : Module S₂ P
                         inst✝¹² : Module R M₂
                         inst✝¹¹ : Module S N₂
                         inst✝¹⁰ : Module R P₂
                         inst✝⁹ : Module S₂ P₂
                         inst✝⁸ : Module R Pₗ
                         inst✝⁷ : Module S Pₗ
                         inst✝⁶ : Module R M'
                         inst✝⁵ : Module R₂ P'
                         inst✝⁴ : Module S₂ P'
                         inst✝³ : SMulCommClass S₂ R₂ P
                         inst✝² : SMulCommClass S R Pₗ
                         inst✝¹ : SMulCommClass S₂ R₂ P'
                         inst✝ : SMulCommClass S₂ R P₂
                         ρ₁₂ : RingHom R R₂
                         σ₁₂ : RingHom S S₂
                         f : LinearMap ρ₁₂ M (LinearMap σ₁₂ N P)
                         n : N
                         m₁ m₂ : M
                         ⊢ Eq ((fun n m => (f m) n) n (HAdd.hAdd m₁ m₂)) (HAdd.hAdd ((fun n m => (f m)  …
                       -/
    (fun n m₁ m₂ => by simp only [map_add, add_apply])
                       /-
                         🎉 no goals
                       -/
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 changed `map_smulₛₗ` into `map_smulₛₗ _`.
    -- It looks like we now run out of assignable metavariables.
                       /-
                         R : Type u_1
                         inst✝²⁹ : Semiring R
                         S : Type u_2
                         inst✝²⁸ : Semiring S
                         R₂ : Type u_3
                         inst✝²⁷ : Semiring R₂
                         S₂ : Type u_4
                         inst✝²⁶ : Semiring S₂
                         M : Type u_5
                         N : Type u_6
                         P : Type u_7
                         M₂ : Type u_8
                         N₂ : Type u_9
                         P₂ : Type u_10
                         Pₗ : Type u_11
                         M' : Type u_12
                         P' : Type u_13
                         inst✝²⁵ : AddCommMonoid M
                         inst✝²⁴ : AddCommMonoid N
                         inst✝²³ : AddCommMonoid P
                         inst✝²² : AddCommMonoid M₂
                         inst✝²¹ : AddCommMonoid N₂
                         inst✝²⁰ : AddCommMonoid P₂
                         inst✝¹⁹ : AddCommMonoid Pₗ
                         inst✝¹⁸ : AddCommGroup M'
                         inst✝¹⁷ : AddCommGroup P'
                         inst✝¹⁶ : Module R M
                         inst✝¹⁵ : Module S N
                         inst✝¹⁴ : Module R₂ P
                         inst✝¹³ : Module S₂ P
                         inst✝¹² : Module R M₂
                         inst✝¹¹ : Module S N₂
                         inst✝¹⁰ : Module R P₂
                         inst✝⁹ : Module S₂ P₂
                         inst✝⁸ : Module R Pₗ
                         inst✝⁷ : Module S Pₗ
                         inst✝⁶ : Module R M'
                         inst✝⁵ : Module R₂ P'
                         inst✝⁴ : Module S₂ P'
                         inst✝³ : SMulCommClass S₂ R₂ P
                         inst✝² : SMulCommClass S R Pₗ
                         inst✝¹ : SMulCommClass S₂ R₂ P'
                         inst✝ : SMulCommClass S₂ R P₂
                         ρ₁₂ : RingHom R R₂
                         σ₁₂ : RingHom S S₂
                         f : LinearMap ρ₁₂ M (LinearMap σ₁₂ N P)
                         c : R
                         n : N
                         m : M
                         ⊢ Eq ((fun n m => (f m) n) n (HSMul.hSMul c m)) (HSMul.hSMul (ρ₁₂ c) ((fun n m …
                       -/
    (fun c n  m  => by simp only [map_smulₛₗ _, smul_apply])
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem flip_apply (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (m : M) (n : N) : flip f n m = f m n := rfl


@[simp]
theorem flip_flip (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) : f.flip.flip = f :=
  LinearMap.ext₂ fun _x _y => (f.flip.flip_apply _ _).trans (f.flip_apply _ _)


theorem flip_inj {f g : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P} (H : flip f = flip g) : f = g :=
                                                  /-
                                                    R : Type u_1
                                                    inst✝¹¹ : Semiring R
                                                    S : Type u_2
                                                    inst✝¹⁰ : Semiring S
                                                    R₂ : Type u_3
                                                    inst✝⁹ : Semiring R₂
                                                    S₂ : Type u_4
                                                    inst✝⁸ : Semiring S₂
                                                    M : Type u_5
                                                    N : Type u_6
                                                    P : Type u_7
                                                    inst✝⁷ : AddCommMonoid M
                                                    inst✝⁶ : AddCommMonoid N
                                                    inst✝⁵ : AddCommMonoid P
                                                    inst✝⁴ : Module R M
                                                    inst✝³ : Module S N
                                                    inst✝² : Module R₂ P
                                                    inst✝¹ : Module S₂ P
                                                    inst✝ : SMulCommClass S₂ R₂ P
                                                    ρ₁₂ : RingHom R R₂
                                                    σ₁₂ : RingHom S S₂
                                                    f g : LinearMap ρ₁₂ M (LinearMap σ₁₂ N P)
                                                    H : Eq f.flip g.flip
                                                    m : M
                                                    n : N
                                                    ⊢ Eq ((f.flip n) m) ((g.flip n) m)
                                                  -/
  ext₂ fun m n => show flip f n m = flip g n m by rw [H]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem map_zero₂ (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (y) : f 0 y = 0 :=
  (flip f y).map_zero


theorem map_neg₂ (f : M' →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P') (x y) : f (-x) y = -f x y :=
  (flip f y).map_neg _


theorem map_sub₂ (f : M' →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P') (x y z) : f (x - y) z = f x z - f y z :=
  (flip f z).map_sub _ _


theorem map_add₂ (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (x₁ x₂ y) : f (x₁ + x₂) y = f x₁ y + f x₂ y :=
  (flip f y).map_add _ _


theorem map_smul₂ (f : M₂ →ₗ[R] N₂ →ₛₗ[σ₁₂] P₂) (r : R) (x y) : f (r • x) y = r • f x y :=
  (flip f y).map_smul _ _


theorem map_smulₛₗ₂ (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (r : R) (x y) : f (r • x) y = ρ₁₂ r • f x y :=
  (flip f y).map_smulₛₗ _ _


theorem map_sum₂ {ι : Type*} (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (t : Finset ι) (x : ι → M) (y) :
    f (∑ i ∈ t, x i) y = ∑ i ∈ t, f (x i) y :=
  _root_.map_sum (flip f y) _ _


/-- Restricting a bilinear map in the second entry -/
def domRestrict₂ (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (q : Submodule S N) : M →ₛₗ[ρ₁₂] q →ₛₗ[σ₁₂] P where
  toFun m := (f m).domRestrict q
                                              /-
                                                R : Type u_1
                                                inst✝²⁹ : Semiring R
                                                S : Type u_2
                                                inst✝²⁸ : Semiring S
                                                R₂ : Type u_3
                                                inst✝²⁷ : Semiring R₂
                                                S₂ : Type u_4
                                                inst✝²⁶ : Semiring S₂
                                                M : Type u_5
                                                N : Type u_6
                                                P : Type u_7
                                                M₂ : Type u_8
                                                N₂ : Type u_9
                                                P₂ : Type u_10
                                                Pₗ : Type u_11
                                                M' : Type u_12
                                                P' : Type u_13
                                                inst✝²⁵ : AddCommMonoid M
                                                inst✝²⁴ : AddCommMonoid N
                                                inst✝²³ : AddCommMonoid P
                                                inst✝²² : AddCommMonoid M₂
                                                inst✝²¹ : AddCommMonoid N₂
                                                inst✝²⁰ : AddCommMonoid P₂
                                                inst✝¹⁹ : AddCommMonoid Pₗ
                                                inst✝¹⁸ : AddCommGroup M'
                                                inst✝¹⁷ : AddCommGroup P'
                                                inst✝¹⁶ : Module R M
                                                inst✝¹⁵ : Module S N
                                                inst✝¹⁴ : Module R₂ P
                                                inst✝¹³ : Module S₂ P
                                                inst✝¹² : Module R M₂
                                                inst✝¹¹ : Module S N₂
                                                inst✝¹⁰ : Module R P₂
                                                inst✝⁹ : Module S₂ P₂
                                                inst✝⁸ : Module R Pₗ
                                                inst✝⁷ : Module S Pₗ
                                                inst✝⁶ : Module R M'
                                                inst✝⁵ : Module R₂ P'
                                                inst✝⁴ : Module S₂ P'
                                                inst✝³ : SMulCommClass S₂ R₂ P
                                                inst✝² : SMulCommClass S R Pₗ
                                                inst✝¹ : SMulCommClass S₂ R₂ P'
                                                inst✝ : SMulCommClass S₂ R P₂
                                                ρ₁₂ : RingHom R R₂
                                                σ₁₂ : RingHom S S₂
                                                f : LinearMap ρ₁₂ M (LinearMap σ₁₂ N P)
                                                q : Submodule S N
                                                m₁ m₂ : M
                                                x✝ : Subtype fun x => Membership.mem q x
                                                ⊢ Eq (((fun m => (f m).domRestrict q) (HAdd.hAdd m₁ m₂)) x✝) ((HAdd.hAdd ((fun …
                                              -/
  map_add' m₁ m₂ := LinearMap.ext fun _ => by simp only [map_add, domRestrict_apply, add_apply]
                                              /-
                                                🎉 no goals
                                              -/
  map_smul' c m :=
                              /-
                                R : Type u_1
                                inst✝²⁹ : Semiring R
                                S : Type u_2
                                inst✝²⁸ : Semiring S
                                R₂ : Type u_3
                                inst✝²⁷ : Semiring R₂
                                S₂ : Type u_4
                                inst✝²⁶ : Semiring S₂
                                M : Type u_5
                                N : Type u_6
                                P : Type u_7
                                M₂ : Type u_8
                                N₂ : Type u_9
                                P₂ : Type u_10
                                Pₗ : Type u_11
                                M' : Type u_12
                                P' : Type u_13
                                inst✝²⁵ : AddCommMonoid M
                                inst✝²⁴ : AddCommMonoid N
                                inst✝²³ : AddCommMonoid P
                                inst✝²² : AddCommMonoid M₂
                                inst✝²¹ : AddCommMonoid N₂
                                inst✝²⁰ : AddCommMonoid P₂
                                inst✝¹⁹ : AddCommMonoid Pₗ
                                inst✝¹⁸ : AddCommGroup M'
                                inst✝¹⁷ : AddCommGroup P'
                                inst✝¹⁶ : Module R M
                                inst✝¹⁵ : Module S N
                                inst✝¹⁴ : Module R₂ P
                                inst✝¹³ : Module S₂ P
                                inst✝¹² : Module R M₂
                                inst✝¹¹ : Module S N₂
                                inst✝¹⁰ : Module R P₂
                                inst✝⁹ : Module S₂ P₂
                                inst✝⁸ : Module R Pₗ
                                inst✝⁷ : Module S Pₗ
                                inst✝⁶ : Module R M'
                                inst✝⁵ : Module R₂ P'
                                inst✝⁴ : Module S₂ P'
                                inst✝³ : SMulCommClass S₂ R₂ P
                                inst✝² : SMulCommClass S R Pₗ
                                inst✝¹ : SMulCommClass S₂ R₂ P'
                                inst✝ : SMulCommClass S₂ R P₂
                                ρ₁₂ : RingHom R R₂
                                σ₁₂ : RingHom S S₂
                                f : LinearMap ρ₁₂ M (LinearMap σ₁₂ N P)
                                q : Submodule S N
                                c : R
                                m : M
                                x✝ : Subtype fun x => Membership.mem q x
                                ⊢ Eq (({ toFun := fun m => (f m).domRestrict q, map_add' := ⋯ }.toFun (HSMul.h …
                              -/
    LinearMap.ext fun _ => by simp only [f.map_smulₛₗ, domRestrict_apply, smul_apply]
                              /-
                                🎉 no goals
                              -/


theorem domRestrict₂_apply (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (q : Submodule S N) (x : M) (y : q) :
    f.domRestrict₂ q x y = f x y := rfl


/-- Restricting a bilinear map in both components -/
def domRestrict₁₂ (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (p : Submodule R M) (q : Submodule S N) :
    p →ₛₗ[ρ₁₂] q →ₛₗ[σ₁₂] P :=
  (f.domRestrict p).domRestrict₂ q


theorem domRestrict₁₂_apply (f : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P) (p : Submodule R M) (q : Submodule S N)
    (x : p) (y : q) : f.domRestrict₁₂ p q x y = f x y := rfl


/-- If `B : M → N → Pₗ` is `R`-`S` bilinear and `R'` and `S'` are compatible scalar multiplications,
then the restriction of scalars is a `R'`-`S'` bilinear map. -/
@[simps!]
def restrictScalars₁₂ (B : M →ₗ[R] N →ₗ[S] Pₗ) : M →ₗ[R'] N →ₗ[S'] Pₗ :=
  LinearMap.mk₂' R' S'
    (B · ·)
    B.map_add₂
    (fun r' m _ ↦ by
      /-
        R : Type u_1
        inst✝⁴² : Semiring R
        S : Type u_2
        inst✝⁴¹ : Semiring S
        R₂ : Type u_3
        inst✝⁴⁰ : Semiring R₂
        S₂ : Type u_4
        inst✝³⁹ : Semiring S₂
        M : Type u_5
        N : Type u_6
        P : Type u_7
        M₂ : Type u_8
        N₂ : Type u_9
        P₂ : Type u_10
        Pₗ : Type u_11
        M' : Type u_12
        P' : Type u_13
        inst✝³⁸ : AddCommMonoid M
        inst✝³⁷ : AddCommMonoid N
        inst✝³⁶ : AddCommMonoid P
        inst✝³⁵ : AddCommMonoid M₂
        inst✝³⁴ : AddCommMonoid N₂
        inst✝³³ : AddCommMonoid P₂
        inst✝³² : AddCommMonoid Pₗ
        inst✝³¹ : AddCommGroup M'
        inst✝³⁰ : AddCommGroup P'
        inst✝²⁹ : Module R M
        inst✝²⁸ : Module S N
        inst✝²⁷ : Module R₂ P
        inst✝²⁶ : Module S₂ P
        inst✝²⁵ : Module R M₂
        inst✝²⁴ : Module S N₂
        inst✝²³ : Module R P₂
        inst✝²² : Module S₂ P₂
        inst✝²¹ : Module R Pₗ
        inst✝²⁰ : Module S Pₗ
        inst✝¹⁹ : Module R M'
        inst✝¹⁸ : Module R₂ P'
        inst✝¹⁷ : Module S₂ P'
        inst✝¹⁶ : SMulCommClass S₂ R₂ P
        inst✝¹⁵ : SMulCommClass S R Pₗ
        inst✝¹⁴ : SMulCommClass S₂ R₂ P'
        inst✝¹³ : SMulCommClass S₂ R P₂
        ρ₁₂ : RingHom R R₂
        σ₁₂ : RingHom S S₂
        R' : Type u_14
        S' : Type u_15
        inst✝¹² : Semiring R'
        inst✝¹¹ : Semiring S'
        inst✝¹⁰ : Module R' M
        inst✝⁹ : Module S' N
        inst✝⁸ : Module R' Pₗ
        inst✝⁷ : Module S' Pₗ
        inst✝⁶ : SMulCommClass S' R' Pₗ
        inst✝⁵ : SMul S' S
        inst✝⁴ : IsScalarTower S' S N
        inst✝³ : IsScalarTower S' S Pₗ
        inst✝² : SMul R' R
        inst✝¹ : IsScalarTower R' R M
        inst✝ : IsScalarTower R' R Pₗ
        B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id S) N Pₗ)
        r' : R'
        m : M
        x✝ : N
        ⊢ Eq ((fun x1 x2 => (B x1) x2) (HSMul.hSMul r' m) x✝) (HSMul.hSMul r' ((fun x1 …
      -/
      dsimp only
      /-
        R : Type u_1
        inst✝⁴² : Semiring R
        S : Type u_2
        inst✝⁴¹ : Semiring S
        R₂ : Type u_3
        inst✝⁴⁰ : Semiring R₂
        S₂ : Type u_4
        inst✝³⁹ : Semiring S₂
        M : Type u_5
        N : Type u_6
        P : Type u_7
        M₂ : Type u_8
        N₂ : Type u_9
        P₂ : Type u_10
        Pₗ : Type u_11
        M' : Type u_12
        P' : Type u_13
        inst✝³⁸ : AddCommMonoid M
        inst✝³⁷ : AddCommMonoid N
        inst✝³⁶ : AddCommMonoid P
        inst✝³⁵ : AddCommMonoid M₂
        inst✝³⁴ : AddCommMonoid N₂
        inst✝³³ : AddCommMonoid P₂
        inst✝³² : AddCommMonoid Pₗ
        inst✝³¹ : AddCommGroup M'
        inst✝³⁰ : AddCommGroup P'
        inst✝²⁹ : Module R M
        inst✝²⁸ : Module S N
        inst✝²⁷ : Module R₂ P
        inst✝²⁶ : Module S₂ P
        inst✝²⁵ : Module R M₂
        inst✝²⁴ : Module S N₂
        inst✝²³ : Module R P₂
        inst✝²² : Module S₂ P₂
        inst✝²¹ : Module R Pₗ
        inst✝²⁰ : Module S Pₗ
        inst✝¹⁹ : Module R M'
        inst✝¹⁸ : Module R₂ P'
        inst✝¹⁷ : Module S₂ P'
        inst✝¹⁶ : SMulCommClass S₂ R₂ P
        inst✝¹⁵ : SMulCommClass S R Pₗ
        inst✝¹⁴ : SMulCommClass S₂ R₂ P'
        inst✝¹³ : SMulCommClass S₂ R P₂
        ρ₁₂ : RingHom R R₂
        σ₁₂ : RingHom S S₂
        R' : Type u_14
        S' : Type u_15
        inst✝¹² : Semiring R'
        inst✝¹¹ : Semiring S'
        inst✝¹⁰ : Module R' M
        inst✝⁹ : Module S' N
        inst✝⁸ : Module R' Pₗ
        inst✝⁷ : Module S' Pₗ
        inst✝⁶ : SMulCommClass S' R' Pₗ
        inst✝⁵ : SMul S' S
        inst✝⁴ : IsScalarTower S' S N
        inst✝³ : IsScalarTower S' S Pₗ
        inst✝² : SMul R' R
        inst✝¹ : IsScalarTower R' R M
        inst✝ : IsScalarTower R' R Pₗ
        B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id S) N Pₗ)
        r' : R'
        m : M
        x✝ : N
        ⊢ Eq ((B (HSMul.hSMul r' m)) x✝) (HSMul.hSMul r' ((B m) x✝))
      -/
      rw [← smul_one_smul R r' m, map_smul₂, smul_one_smul])
      /-
        🎉 no goals
      -/
    (fun _ ↦ map_add _)
    (fun _ x ↦ (B x).map_smul_of_tower _)


theorem restrictScalars₁₂_injective : Function.Injective
    (LinearMap.restrictScalars₁₂ R' S' : (M →ₗ[R] N →ₗ[S] Pₗ) → (M →ₗ[R'] N →ₗ[S'] Pₗ)) :=
  fun _ _ h ↦ ext₂ (congr_fun₂ h : _)


@[simp]
theorem restrictScalars₁₂_inj {B B' : M →ₗ[R] N →ₗ[S] Pₗ} :
    B.restrictScalars₁₂ R' S' = B'.restrictScalars₁₂ R' S' ↔ B = B' :=
  (restrictScalars₁₂_injective R' S').eq_iff


/-- Create a bilinear map from a function that is linear in each component.

This is a shorthand for `mk₂'` for the common case when `R = S`. -/
def mk₂ (f : M → Nₗ → Pₗ) (H1 : ∀ m₁ m₂ n, f (m₁ + m₂) n = f m₁ n + f m₂ n)
    (H2 : ∀ (c : R) (m n), f (c • m) n = c • f m n)
    (H3 : ∀ m n₁ n₂, f m (n₁ + n₂) = f m n₁ + f m n₂)
    (H4 : ∀ (c : R) (m n), f m (c • n) = c • f m n) : M →ₗ[R] Nₗ →ₗ[R] Pₗ :=
  mk₂' R R f H1 H2 H3 H4


@[simp]
theorem mk₂_apply (f : M → Nₗ → Pₗ) {H1 H2 H3 H4} (m : M) (n : Nₗ) :
    (mk₂ R f H1 H2 H3 H4 : M →ₗ[R] Nₗ →ₗ[R] Pₗ) m n = f m n := rfl


/-- Given a linear map from `M` to linear maps from `N` to `P`, i.e., a bilinear map `M → N → P`,
change the order of variables and get a linear map from `N` to linear maps from `M` to `P`. -/
def lflip : (M →ₛₗ[σ₁₃] N →ₛₗ[σ₂₃] P) →ₗ[R₃] N →ₛₗ[σ₂₃] M →ₛₗ[σ₁₃] P where
  toFun := flip
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem lflip_apply (m : M) (n : N) : lflip f n m = f m n := rfl


/-- Composing a given linear map `M → N` with a linear map `N → P` as a linear map from
`Nₗ →ₗ[R] Pₗ` to `M →ₗ[R] Pₗ`. -/
def lcomp (f : M →ₗ[R] Nₗ) : (Nₗ →ₗ[R] Pₗ) →ₗ[R] M →ₗ[R] Pₗ :=
  flip <| LinearMap.comp (flip id) f


@[simp]
theorem lcomp_apply (f : M →ₗ[R] Nₗ) (g : Nₗ →ₗ[R] Pₗ) (x : M) : lcomp _ _ f g x = g (f x) := rfl


theorem lcomp_apply' (f : M →ₗ[R] Nₗ) (g : Nₗ →ₗ[R] Pₗ) : lcomp R Pₗ f g = g ∘ₗ f := rfl


/-- Composing a semilinear map `M → N` and a semilinear map `N → P` to form a semilinear map
`M → P` is itself a linear map. -/
def lcompₛₗ (f : M →ₛₗ[σ₁₂] N) : (N →ₛₗ[σ₂₃] P) →ₗ[R₃] M →ₛₗ[σ₁₃] P :=
  flip <| LinearMap.comp (flip id) f


@[simp]
theorem lcompₛₗ_apply (f : M →ₛₗ[σ₁₂] N) (g : N →ₛₗ[σ₂₃] P) (x : M) :
    lcompₛₗ P σ₂₃ f g x = g (f x) := rfl


/-- Composing linear maps as a bilinear map from `(M →ₗ[R] N) × (N →ₗ[R] P)` to `M →ₗ[R] P` -/
def llcomp : (Nₗ →ₗ[R] Pₗ) →ₗ[R] (M →ₗ[R] Nₗ) →ₗ[R] M →ₗ[R] Pₗ :=
  flip
    { toFun := lcomp R Pₗ
      map_add' := fun _f _f' => ext₂ fun g _x => g.map_add _ _
      map_smul' := fun (_c : R) _f => ext₂ fun g _x => g.map_smul _ _ }


@[simp]
theorem llcomp_apply (f : Nₗ →ₗ[R] Pₗ) (g : M →ₗ[R] Nₗ) (x : M) :
    llcomp R M Nₗ Pₗ f g x = f (g x) := rfl


theorem llcomp_apply' (f : Nₗ →ₗ[R] Pₗ) (g : M →ₗ[R] Nₗ) : llcomp R M Nₗ Pₗ f g = f ∘ₗ g := rfl


/-- Composing a linear map `Q → N` and a bilinear map `M → N → P` to
form a bilinear map `M → Q → P`. -/
def compl₂ {R₅ : Type*} [CommSemiring R₅] [Module R₅ P] [SMulCommClass R₃ R₅ P] {σ₁₅ : R →+* R₅}
    (h : M →ₛₗ[σ₁₅] N →ₛₗ[σ₂₃] P) (g : Q →ₛₗ[σ₄₂] N) : M →ₛₗ[σ₁₅] Q →ₛₗ[σ₄₃] P where
  toFun a := (lcompₛₗ P σ₂₃ g) (h a)
  map_add' _ _ := by
    /-
      R : Type u_1
      inst✝²⁶ : CommSemiring R
      R₂ : Type u_2
      inst✝²⁵ : CommSemiring R₂
      R₃ : Type u_3
      inst✝²⁴ : CommSemiring R₃
      R₄ : Type u_4
      inst✝²³ : CommSemiring R₄
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝²² : AddCommMonoid M
      inst✝²¹ : AddCommMonoid N
      inst✝²⁰ : AddCommMonoid P
      inst✝¹⁹ : AddCommMonoid Q
      inst✝¹⁸ : AddCommMonoid Mₗ
      inst✝¹⁷ : AddCommMonoid Nₗ
      inst✝¹⁶ : AddCommMonoid Pₗ
      inst✝¹⁵ : AddCommMonoid Qₗ
      inst✝¹⁴ : AddCommMonoid Qₗ'
      inst✝¹³ : Module R M
      inst✝¹² : Module R₂ N
      inst✝¹¹ : Module R₃ P
      inst✝¹⁰ : Module R₄ Q
      inst✝⁹ : Module R Mₗ
      inst✝⁸ : Module R Nₗ
      inst✝⁷ : Module R Pₗ
      inst✝⁶ : Module R Qₗ
      inst✝⁵ : Module R Qₗ'
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      σ₄₂ : RingHom R₄ R₂
      σ₄₃ : RingHom R₄ R₃
      inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝³ : RingHomCompTriple σ₄₂ σ₂₃ σ₄₃
      f : LinearMap σ₁₃ M (LinearMap σ₂₃ N P)
      R₅ : Type u_14
      inst✝² : CommSemiring R₅
      inst✝¹ : Module R₅ P
      inst✝ : SMulCommClass R₃ R₅ P
      σ₁₅ : RingHom R R₅
      h : LinearMap σ₁₅ M (LinearMap σ₂₃ N P)
      g : LinearMap σ₄₂ Q N
      x✝¹ x✝ : M
      ⊢ Eq ((fun a => (LinearMap.lcompₛₗ P σ₂₃ g) (h a)) (HAdd.hAdd x✝¹ x✝)) (HAdd.h …
    -/
    simp [map_add]
    /-
      🎉 no goals
    -/
  map_smul' _ _ := by
    /-
      R : Type u_1
      inst✝²⁶ : CommSemiring R
      R₂ : Type u_2
      inst✝²⁵ : CommSemiring R₂
      R₃ : Type u_3
      inst✝²⁴ : CommSemiring R₃
      R₄ : Type u_4
      inst✝²³ : CommSemiring R₄
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝²² : AddCommMonoid M
      inst✝²¹ : AddCommMonoid N
      inst✝²⁰ : AddCommMonoid P
      inst✝¹⁹ : AddCommMonoid Q
      inst✝¹⁸ : AddCommMonoid Mₗ
      inst✝¹⁷ : AddCommMonoid Nₗ
      inst✝¹⁶ : AddCommMonoid Pₗ
      inst✝¹⁵ : AddCommMonoid Qₗ
      inst✝¹⁴ : AddCommMonoid Qₗ'
      inst✝¹³ : Module R M
      inst✝¹² : Module R₂ N
      inst✝¹¹ : Module R₃ P
      inst✝¹⁰ : Module R₄ Q
      inst✝⁹ : Module R Mₗ
      inst✝⁸ : Module R Nₗ
      inst✝⁷ : Module R Pₗ
      inst✝⁶ : Module R Qₗ
      inst✝⁵ : Module R Qₗ'
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      σ₄₂ : RingHom R₄ R₂
      σ₄₃ : RingHom R₄ R₃
      inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝³ : RingHomCompTriple σ₄₂ σ₂₃ σ₄₃
      f : LinearMap σ₁₃ M (LinearMap σ₂₃ N P)
      R₅ : Type u_14
      inst✝² : CommSemiring R₅
      inst✝¹ : Module R₅ P
      inst✝ : SMulCommClass R₃ R₅ P
      σ₁₅ : RingHom R R₅
      h : LinearMap σ₁₅ M (LinearMap σ₂₃ N P)
      g : LinearMap σ₄₂ Q N
      x✝¹ : R
      x✝ : M
      ⊢ Eq ({ toFun := fun a => (LinearMap.lcompₛₗ P σ₂₃ g) (h a), map_add' := ⋯ }.t …
    -/
    simp only [LinearMap.map_smulₛₗ, lcompₛₗ]
    /-
      R : Type u_1
      inst✝²⁶ : CommSemiring R
      R₂ : Type u_2
      inst✝²⁵ : CommSemiring R₂
      R₃ : Type u_3
      inst✝²⁴ : CommSemiring R₃
      R₄ : Type u_4
      inst✝²³ : CommSemiring R₄
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝²² : AddCommMonoid M
      inst✝²¹ : AddCommMonoid N
      inst✝²⁰ : AddCommMonoid P
      inst✝¹⁹ : AddCommMonoid Q
      inst✝¹⁸ : AddCommMonoid Mₗ
      inst✝¹⁷ : AddCommMonoid Nₗ
      inst✝¹⁶ : AddCommMonoid Pₗ
      inst✝¹⁵ : AddCommMonoid Qₗ
      inst✝¹⁴ : AddCommMonoid Qₗ'
      inst✝¹³ : Module R M
      inst✝¹² : Module R₂ N
      inst✝¹¹ : Module R₃ P
      inst✝¹⁰ : Module R₄ Q
      inst✝⁹ : Module R Mₗ
      inst✝⁸ : Module R Nₗ
      inst✝⁷ : Module R Pₗ
      inst✝⁶ : Module R Qₗ
      inst✝⁵ : Module R Qₗ'
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      σ₄₂ : RingHom R₄ R₂
      σ₄₃ : RingHom R₄ R₃
      inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝³ : RingHomCompTriple σ₄₂ σ₂₃ σ₄₃
      f : LinearMap σ₁₃ M (LinearMap σ₂₃ N P)
      R₅ : Type u_14
      inst✝² : CommSemiring R₅
      inst✝¹ : Module R₅ P
      inst✝ : SMulCommClass R₃ R₅ P
      σ₁₅ : RingHom R R₅
      h : LinearMap σ₁₅ M (LinearMap σ₂₃ N P)
      g : LinearMap σ₄₂ Q N
      x✝¹ : R
      x✝ : M
      ⊢ Eq ((LinearMap.id.flip.comp g).flip (HSMul.hSMul (σ₁₅ x✝¹) (h x✝))) (HSMul.h …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem compl₂_apply (g : Q →ₛₗ[σ₄₂] N) (m : M) (q : Q) : f.compl₂ g m q = f m (g q) := rfl


@[simp]
theorem compl₂_id : f.compl₂ LinearMap.id = f := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    R₂ : Type u_2
    inst✝⁷ : CommSemiring R₂
    R₃ : Type u_3
    inst✝⁶ : CommSemiring R₃
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R₂ N
    inst✝ : Module R₃ P
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    f : LinearMap σ₁₃ M (LinearMap σ₂₃ N P)
    ⊢ Eq (f.compl₂ LinearMap.id) f
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝⁸ : CommSemiring R
    R₂ : Type u_2
    inst✝⁷ : CommSemiring R₂
    R₃ : Type u_3
    inst✝⁶ : CommSemiring R₃
    M : Type u_5
    N : Type u_6
    P : Type u_7
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R₂ N
    inst✝ : Module R₃ P
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    f : LinearMap σ₁₃ M (LinearMap σ₂₃ N P)
    x✝¹ : M
    x✝ : N
    ⊢ Eq (((f.compl₂ LinearMap.id) x✝¹) x✝) ((f x✝¹) x✝)
  -/
  rw [compl₂_apply, id_coe, _root_.id]
  /-
    🎉 no goals
  -/


/-- Composing linear maps `Q → M` and `Q' → N` with a bilinear map `M → N → P` to
form a bilinear map `Q → Q' → P`. -/
def compl₁₂ {R₁ : Type*} [CommSemiring R₁] [Module R₂ N] [Module R₂ Pₗ] [Module R₁ Pₗ]
    [Module R₁ Mₗ] [SMulCommClass R₂ R₁ Pₗ] [Module R₁ Qₗ] [Module R₂ Qₗ']
    (f : Mₗ →ₗ[R₁] N →ₗ[R₂] Pₗ) (g : Qₗ →ₗ[R₁] Mₗ) (g' : Qₗ' →ₗ[R₂] N) :
    Qₗ →ₗ[R₁] Qₗ' →ₗ[R₂] Pₗ :=
  (f.comp g).compl₂ g'


@[simp]
theorem compl₁₂_apply (f : Mₗ →ₗ[R] Nₗ →ₗ[R] Pₗ) (g : Qₗ →ₗ[R] Mₗ) (g' : Qₗ' →ₗ[R] Nₗ) (x : Qₗ)
    (y : Qₗ') : f.compl₁₂ g g' x y = f (g x) (g' y) := rfl


@[simp]
theorem compl₁₂_id_id (f : Mₗ →ₗ[R] Nₗ →ₗ[R] Pₗ) : f.compl₁₂ LinearMap.id LinearMap.id = f := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    Mₗ : Type u_9
    Nₗ : Type u_10
    Pₗ : Type u_11
    inst✝⁵ : AddCommMonoid Mₗ
    inst✝⁴ : AddCommMonoid Nₗ
    inst✝³ : AddCommMonoid Pₗ
    inst✝² : Module R Mₗ
    inst✝¹ : Module R Nₗ
    inst✝ : Module R Pₗ
    f : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
    ⊢ Eq (f.compl₁₂ LinearMap.id LinearMap.id) f
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝⁶ : CommSemiring R
    Mₗ : Type u_9
    Nₗ : Type u_10
    Pₗ : Type u_11
    inst✝⁵ : AddCommMonoid Mₗ
    inst✝⁴ : AddCommMonoid Nₗ
    inst✝³ : AddCommMonoid Pₗ
    inst✝² : Module R Mₗ
    inst✝¹ : Module R Nₗ
    inst✝ : Module R Pₗ
    f : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
    x✝¹ : Mₗ
    x✝ : Nₗ
    ⊢ Eq (((f.compl₁₂ LinearMap.id LinearMap.id) x✝¹) x✝) ((f x✝¹) x✝)
  -/
  simp_rw [compl₁₂_apply, id_coe, _root_.id]
  /-
    🎉 no goals
  -/


theorem compl₁₂_inj {f₁ f₂ : Mₗ →ₗ[R] Nₗ →ₗ[R] Pₗ} {g : Qₗ →ₗ[R] Mₗ} {g' : Qₗ' →ₗ[R] Nₗ}
    (hₗ : Function.Surjective g) (hᵣ : Function.Surjective g') :
    f₁.compl₁₂ g g' = f₂.compl₁₂ g g' ↔ f₁ = f₂ := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    Mₗ : Type u_9
    Nₗ : Type u_10
    Pₗ : Type u_11
    Qₗ : Type u_12
    Qₗ' : Type u_13
    inst✝⁹ : AddCommMonoid Mₗ
    inst✝⁸ : AddCommMonoid Nₗ
    inst✝⁷ : AddCommMonoid Pₗ
    inst✝⁶ : AddCommMonoid Qₗ
    inst✝⁵ : AddCommMonoid Qₗ'
    inst✝⁴ : Module R Mₗ
    inst✝³ : Module R Nₗ
    inst✝² : Module R Pₗ
    inst✝¹ : Module R Qₗ
    inst✝ : Module R Qₗ'
    f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
    g : LinearMap (RingHom.id R) Qₗ Mₗ
    g' : LinearMap (RingHom.id R) Qₗ' Nₗ
    hₗ : Function.Surjective ⇑g
    hᵣ : Function.Surjective ⇑g'
    ⊢ Iff (Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')) (Eq f₁ f₂)
  -/
  constructor <;> intro h
  · -- B₁.comp l r = B₂.comp l r → B₁ = B₂
    /-
      case mp
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
      ⊢ Eq f₁ f₂
    -/
    ext x y
    /-
      case mp.h.h
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
      x : Mₗ
      y : Nₗ
      ⊢ Eq ((f₁ x) y) ((f₂ x) y)
    -/
    cases' hₗ x with x' hx
    /-
      case mp.h.h.intro
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
      x : Mₗ
      y : Nₗ
      x' : Qₗ
      hx : Eq (g x') x
      ⊢ Eq ((f₁ x) y) ((f₂ x) y)
    -/
    subst hx
    /-
      case mp.h.h.intro
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
      y : Nₗ
      x' : Qₗ
      ⊢ Eq ((f₁ (g x')) y) ((f₂ (g x')) y)
    -/
    cases' hᵣ y with y' hy
    /-
      case mp.h.h.intro.intro
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
      y : Nₗ
      x' : Qₗ
      y' : Qₗ'
      hy : Eq (g' y') y
      ⊢ Eq ((f₁ (g x')) y) ((f₂ (g x')) y)
    -/
    subst hy
    /-
      case mp.h.h.intro.intro
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
      x' : Qₗ
      y' : Qₗ'
      ⊢ Eq ((f₁ (g x')) (g' y')) ((f₂ (g x')) (g' y'))
    -/
    convert LinearMap.congr_fun₂ h x' y' using 0
    /-
      🎉 no goals
    -/
  · -- B₁ = B₂ → B₁.comp l r = B₂.comp l r
    /-
      case mpr
      R : Type u_1
      inst✝¹⁰ : CommSemiring R
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝⁹ : AddCommMonoid Mₗ
      inst✝⁸ : AddCommMonoid Nₗ
      inst✝⁷ : AddCommMonoid Pₗ
      inst✝⁶ : AddCommMonoid Qₗ
      inst✝⁵ : AddCommMonoid Qₗ'
      inst✝⁴ : Module R Mₗ
      inst✝³ : Module R Nₗ
      inst✝² : Module R Pₗ
      inst✝¹ : Module R Qₗ
      inst✝ : Module R Qₗ'
      f₁ f₂ : LinearMap (RingHom.id R) Mₗ (LinearMap (RingHom.id R) Nₗ Pₗ)
      g : LinearMap (RingHom.id R) Qₗ Mₗ
      g' : LinearMap (RingHom.id R) Qₗ' Nₗ
      hₗ : Function.Surjective ⇑g
      hᵣ : Function.Surjective ⇑g'
      h : Eq f₁ f₂
      ⊢ Eq (f₁.compl₁₂ g g') (f₂.compl₁₂ g g')
    -/
    subst h; rfl
             /-
               🎉 no goals
             -/


/-- Composing a linear map `P → Q` and a bilinear map `M → N → P` to
form a bilinear map `M → N → Q`. -/
def compr₂ (f : M →ₗ[R] Nₗ →ₗ[R] Pₗ) (g : Pₗ →ₗ[R] Qₗ) : M →ₗ[R] Nₗ →ₗ[R] Qₗ :=
  llcomp R Nₗ Pₗ Qₗ g ∘ₗ f


@[simp]
theorem compr₂_apply (f : M →ₗ[R] Nₗ →ₗ[R] Pₗ) (g : Pₗ →ₗ[R] Qₗ) (m : M) (n : Nₗ) :
    f.compr₂ g m n = g (f m n) := rfl


/-- Scalar multiplication as a bilinear map `R → M → M`. -/
def lsmul : R →ₗ[R] M →ₗ[R] M :=
  mk₂ R (· • ·) add_smul (fun _ _ _ => mul_smul _ _ _) smul_add fun r s m => by
    /-
      R : Type u_1
      inst✝²³ : CommSemiring R
      R₂ : Type u_2
      inst✝²² : CommSemiring R₂
      R₃ : Type u_3
      inst✝²¹ : CommSemiring R₃
      R₄ : Type u_4
      inst✝²⁰ : CommSemiring R₄
      M : Type u_5
      N : Type u_6
      P : Type u_7
      Q : Type u_8
      Mₗ : Type u_9
      Nₗ : Type u_10
      Pₗ : Type u_11
      Qₗ : Type u_12
      Qₗ' : Type u_13
      inst✝¹⁹ : AddCommMonoid M
      inst✝¹⁸ : AddCommMonoid N
      inst✝¹⁷ : AddCommMonoid P
      inst✝¹⁶ : AddCommMonoid Q
      inst✝¹⁵ : AddCommMonoid Mₗ
      inst✝¹⁴ : AddCommMonoid Nₗ
      inst✝¹³ : AddCommMonoid Pₗ
      inst✝¹² : AddCommMonoid Qₗ
      inst✝¹¹ : AddCommMonoid Qₗ'
      inst✝¹⁰ : Module R M
      inst✝⁹ : Module R₂ N
      inst✝⁸ : Module R₃ P
      inst✝⁷ : Module R₄ Q
      inst✝⁶ : Module R Mₗ
      inst✝⁵ : Module R Nₗ
      inst✝⁴ : Module R Pₗ
      inst✝³ : Module R Qₗ
      inst✝² : Module R Qₗ'
      σ₁₂ : RingHom R R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R R₃
      σ₄₂ : RingHom R₄ R₂
      σ₄₃ : RingHom R₄ R₃
      inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝ : RingHomCompTriple σ₄₂ σ₂₃ σ₄₃
      f : LinearMap σ₁₃ M (LinearMap σ₂₃ N P)
      r s : R
      m : M
      ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) s (HSMul.hSMul r m)) (HSMul.hSMul r ((f …
    -/
    simp only [smul_smul, smul_eq_mul, mul_comm]
    /-
      🎉 no goals
    -/


lemma lsmul_eq_DistribMulAction_toLinearMap (r : R) :
    lsmul R M r = DistribMulAction.toLinearMap R M r := rfl


@[simp]
theorem lsmul_apply (r : R) (m : M) : lsmul R M r m = r • m := rfl


variable (R M Nₗ) in
/-- A shorthand for the type of `R`-bilinear `Nₗ`-valued maps on `M`. -/
protected abbrev BilinMap : Type _ := M →ₗ[R] M →ₗ[R] Nₗ


variable (R M) in
/-- For convenience, a shorthand for the type of bilinear forms from `M` to `R`. -/
protected abbrev BilinForm : Type _ := LinearMap.BilinMap R M R


theorem lsmul_injective [NoZeroSMulDivisors R M] {x : R} (hx : x ≠ 0) :
    Function.Injective (lsmul R M x) :=
  smul_right_injective _ hx


theorem ker_lsmul [NoZeroSMulDivisors R M] {a : R} (ha : a ≠ 0) :
    LinearMap.ker (LinearMap.lsmul R M a) = ⊥ :=
  LinearMap.ker_eq_bot_of_injective (LinearMap.lsmul_injective ha)


/-- Restrict the scalars, domains, and range of a bilinear map. -/
noncomputable def restrictScalarsRange :
    M' →ₗ[S] N' →ₗ[S] P' :=
  (((LinearMap.restrictScalarsₗ S R _ _ _).comp
    (B.restrictScalars S)).compl₁₂ i j).codRestrict₂ k hk hB


@[simp] lemma restrictScalarsRange_apply (m : M') (n : N') :
    k (restrictScalarsRange i j k hk B hB m n) = B (i m) (j n) := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    P : Type u_5
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : SMul S R
    inst✝¹⁸ : AddCommMonoid M
    inst✝¹⁷ : Module R M
    inst✝¹⁶ : AddCommMonoid N
    inst✝¹⁵ : Module R N
    inst✝¹⁴ : AddCommMonoid P
    inst✝¹³ : Module R P
    inst✝¹² : Module S M
    inst✝¹¹ : Module S N
    inst✝¹⁰ : Module S P
    inst✝⁹ : IsScalarTower S R M
    inst✝⁸ : IsScalarTower S R N
    inst✝⁷ : IsScalarTower S R P
    inst✝⁶ : AddCommMonoid M'
    inst✝⁵ : Module S M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module S N'
    inst✝² : AddCommMonoid P'
    inst✝¹ : Module S P'
    inst✝ : SMulCommClass R S P
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    k : LinearMap (RingHom.id S) P' P
    hk : Function.Injective ⇑k
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    hB : ∀ (m : M') (n : N'), Membership.mem (LinearMap.range k) ((B (i m)) (j n))
    m : M'
    n : N'
    ⊢ Eq (k (((i.restrictScalarsRange j k hk B hB) m) n)) ((B (i m)) (j n))
  -/
  simp [restrictScalarsRange]
  /-
    🎉 no goals
  -/


@[simp]
lemma restrictScalarsRange_apply_eq_zero_iff (m : M') (n : N') :
    restrictScalarsRange i j k hk B hB m n = 0 ↔ B (i m) (j n) = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    P : Type u_5
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : SMul S R
    inst✝¹⁸ : AddCommMonoid M
    inst✝¹⁷ : Module R M
    inst✝¹⁶ : AddCommMonoid N
    inst✝¹⁵ : Module R N
    inst✝¹⁴ : AddCommMonoid P
    inst✝¹³ : Module R P
    inst✝¹² : Module S M
    inst✝¹¹ : Module S N
    inst✝¹⁰ : Module S P
    inst✝⁹ : IsScalarTower S R M
    inst✝⁸ : IsScalarTower S R N
    inst✝⁷ : IsScalarTower S R P
    inst✝⁶ : AddCommMonoid M'
    inst✝⁵ : Module S M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module S N'
    inst✝² : AddCommMonoid P'
    inst✝¹ : Module S P'
    inst✝ : SMulCommClass R S P
    i : LinearMap (RingHom.id S) M' M
    j : LinearMap (RingHom.id S) N' N
    k : LinearMap (RingHom.id S) P' P
    hk : Function.Injective ⇑k
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
    hB : ∀ (m : M') (n : N'), Membership.mem (LinearMap.range k) ((B (i m)) (j n))
    m : M'
    n : N'
    ⊢ Iff (Eq (((i.restrictScalarsRange j k hk B hB) m) n) 0) (Eq ((B (i m)) (j n) …
  -/
  rw [← hk.eq_iff, restrictScalarsRange_apply, map_zero]
  /-
    🎉 no goals
  -/


