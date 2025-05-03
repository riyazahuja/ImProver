/-- An associative ring gives rise to a Lie ring by taking the bracket to be the ring commutator. -/
instance (priority := 100) ofAssociativeRing : LieRing A where
                      /-
                        A : Type v
                        inst✝ : Ring A
                        x✝² x✝¹ x✝ : A
                        ⊢ Eq (Bracket.bracket (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (Bracket.bracket x✝²  …
                      -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  add_lie _ _ _ := by simp only [Ring.lie_def, right_distrib, left_distrib]; abel
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                      /-
                        A : Type v
                        inst✝ : Ring A
                        x✝² x✝¹ x✝ : A
                        ⊢ Eq (Bracket.bracket x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (Bracket.bracket x✝²  …
                      -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  lie_add _ _ _ := by simp only [Ring.lie_def, right_distrib, left_distrib]; abel
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                 /-
                   A : Type v
                   inst✝ : Ring A
                   ⊢ ∀ (x : A), Eq (Bracket.bracket x x) 0
                 -/
  lie_self := by simp only [Ring.lie_def, forall_const, sub_self]
                 /-
                   🎉 no goals
                 -/
  leibniz_lie _ _ _ := by
    /-
      A : Type v
      inst✝ : Ring A
      x✝² x✝¹ x✝ : A
      ⊢ Eq (Bracket.bracket x✝² (Bracket.bracket x✝¹ x✝)) (HAdd.hAdd (Bracket.bracke …
    -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    simp only [Ring.lie_def, mul_sub_left_distrib, mul_sub_right_distrib, mul_assoc]; abel
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem of_associative_ring_bracket (x y : A) : ⁅x, y⁆ = x * y - y * x :=
  rfl


@[simp]
theorem lie_apply {α : Type*} (f g : α → A) (a : α) : ⁅f, g⁆ a = ⁅f a, g a⁆ :=
  rfl


/-- We can regard a module over an associative ring `A` as a Lie ring module over `A` with Lie
bracket equal to its ring commutator.

Note that this cannot be a global instance because it would create a diamond when `M = A`,
specifically we can build two mathematically-different `bracket A A`s:
 1. `@Ring.bracket A _` which says `⁅a, b⁆ = a * b - b * a`
 2. `(@LieRingModule.ofAssociativeModule A _ A _ _).toBracket` which says `⁅a, b⁆ = a • b`
    (and thus `⁅a, b⁆ = a * b`)

See note [reducible non-instances] -/
abbrev LieRingModule.ofAssociativeModule : LieRingModule A M where
  bracket := (· • ·)
  add_lie := add_smul
  lie_add := smul_add
                    /-
                      A : Type v
                      inst✝² : Ring A
                      M : Type w
                      inst✝¹ : AddCommGroup M
                      inst✝ : Module A M
                      ⊢ ∀ (x y : A) (m : M), Eq (Bracket.bracket x (Bracket.bracket y m)) (HAdd.hAdd …
                    -/
  leibniz_lie := by simp [LieRing.of_associative_ring_bracket, sub_smul, mul_smul, sub_add_cancel]
                    /-
                      🎉 no goals
                    -/


theorem lie_eq_smul (a : A) (m : M) : ⁅a, m⁆ = a • m :=
  rfl


/-- An associative algebra gives rise to a Lie algebra by taking the bracket to be the ring
commutator. -/
instance (priority := 100) LieAlgebra.ofAssociativeAlgebra : LieAlgebra R A where
  lie_smul t x y := by
    rw [LieRing.of_associative_ring_bracket, LieRing.of_associative_ring_bracket,
      Algebra.mul_smul_comm, Algebra.smul_mul_assoc, smul_sub]


/-- A representation of an associative algebra `A` is also a representation of `A`, regarded as a
Lie algebra via the ring commutator.

See the comment at `LieRingModule.ofAssociativeModule` for why the possibility `M = A` means
this cannot be a global instance. -/
theorem LieModule.ofAssociativeModule : LieModule R A M where
  smul_lie := smul_assoc
  lie_smul := smul_algebra_smul_comm


instance Module.End.instLieRingModule : LieRingModule (Module.End R M) M :=
  LieRingModule.ofAssociativeModule


instance Module.End.instLieModule : LieModule R (Module.End R M) M :=
  LieModule.ofAssociativeModule


@[simp] lemma Module.End.lie_apply (f : Module.End R M) (m : M) : ⁅f, m⁆ = f m := rfl


/-- The map `ofAssociativeAlgebra` associating a Lie algebra to an associative algebra is
functorial. -/
def toLieHom : A →ₗ⁅R⁆ B :=
  { f.toLinearMap with
                                /-
                                  A : Type v
                                  inst✝⁶ : Ring A
                                  R : Type u
                                  inst✝⁵ : CommRing R
                                  inst✝⁴ : Algebra R A
                                  B : Type w
                                  C : Type w₁
                                  inst✝³ : Ring B
                                  inst✝² : Ring C
                                  inst✝¹ : Algebra R B
                                  inst✝ : Algebra R C
                                  f : AlgHom R A B
                                  g : AlgHom R B C
                                  x✝¹ x✝ : A
                                  ⊢ Eq (__src✝.toFun (Bracket.bracket x✝¹ x✝)) (Bracket.bracket (__src✝.toFun x✝ …
                                -/
    map_lie' := fun {_ _} => by simp [LieRing.of_associative_ring_bracket] }
                                /-
                                  🎉 no goals
                                -/


instance : Coe (A →ₐ[R] B) (A →ₗ⁅R⁆ B) :=
  ⟨toLieHom⟩

/- Porting note: is a syntactic tautology
@[simp]
theorem toLieHom_coe : f.toLieHom = ↑f :=
  rfl
-/


@[simp]
theorem coe_toLieHom : ((f : A →ₗ⁅R⁆ B) : A → B) = f :=
  rfl


theorem toLieHom_apply (x : A) : f.toLieHom x = f x :=
  rfl


@[simp]
theorem toLieHom_id : (AlgHom.id R A : A →ₗ⁅R⁆ A) = LieHom.id :=
  rfl


@[simp]
theorem toLieHom_comp : (g.comp f : A →ₗ⁅R⁆ C) = (g : B →ₗ⁅R⁆ C).comp (f : A →ₗ⁅R⁆ B) :=
  rfl


theorem toLieHom_injective {f g : A →ₐ[R] B} (h : (f : A →ₗ⁅R⁆ B) = (g : A →ₗ⁅R⁆ B)) : f = g := by
  /-
    A : Type v
    inst✝⁴ : Ring A
    R : Type u
    inst✝³ : CommRing R
    inst✝² : Algebra R A
    B : Type w
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    f g : AlgHom R A B
    h : Eq f.toLieHom g.toLieHom
    ⊢ Eq f g
  -/
  ext a; exact LieHom.congr_fun h a
         /-
           🎉 no goals
         -/


/-- A Lie module yields a Lie algebra morphism into the linear endomorphisms of the module.

See also `LieModule.toModuleHom`. -/
@[simps]
def LieModule.toEnd : L →ₗ⁅R⁆ Module.End R M where
  toFun x :=
    { toFun := fun m => ⁅x, m⁆
      map_add' := lie_add x
      map_smul' := fun t => lie_smul t x }
                     /-
                       R : Type u
                       L : Type v
                       M : Type w
                       inst✝⁶ : CommRing R
                       inst✝⁵ : LieRing L
                       inst✝⁴ : LieAlgebra R L
                       inst✝³ : AddCommGroup M
                       inst✝² : Module R M
                       inst✝¹ : LieRingModule L M
                       inst✝ : LieModule R L M
                       x y : L
                       ⊢ Eq ((fun x => { toFun := fun m => Bracket.bracket x m, map_add' := ⋯, map_sm …
                     -/
  map_add' x y := by ext m; apply add_lie
                            /-
                              🎉 no goals
                            -/
                      /-
                        R : Type u
                        L : Type v
                        M : Type w
                        inst✝⁶ : CommRing R
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        t : R
                        x : L
                        ⊢ Eq ({ toFun := fun x => { toFun := fun m => Bracket.bracket x m, map_add' := …
                      -/
  map_smul' t x := by ext m; apply smul_lie
                             /-
                               🎉 no goals
                             -/
                       /-
                         R : Type u
                         L : Type v
                         M : Type w
                         inst✝⁶ : CommRing R
                         inst✝⁵ : LieRing L
                         inst✝⁴ : LieAlgebra R L
                         inst✝³ : AddCommGroup M
                         inst✝² : Module R M
                         inst✝¹ : LieRingModule L M
                         inst✝ : LieModule R L M
                         x y : L
                         ⊢ Eq ({ toFun := fun x => { toFun := fun m => Bracket.bracket x m, map_add' := …
                       -/
  map_lie' {x y} := by ext m; apply lie_lie
                              /-
                                🎉 no goals
                              -/


/-- The adjoint action of a Lie algebra on itself. -/
def LieAlgebra.ad : L →ₗ⁅R⁆ Module.End R L :=
  LieModule.toEnd R L L


@[simp]
theorem LieAlgebra.ad_apply (x y : L) : LieAlgebra.ad R L x y = ⁅x, y⁆ :=
  rfl


@[simp]
theorem LieModule.toEnd_module_end :
                                                           /-
                                                             R : Type u
                                                             M : Type w
                                                             inst✝² : CommRing R
                                                             inst✝¹ : AddCommGroup M
                                                             inst✝ : Module R M
                                                             ⊢ Eq (LieModule.toEnd R (Module.End R M) M) LieHom.id
                                                           -/
    LieModule.toEnd R (Module.End R M) M = LieHom.id := by ext g m; simp [lie_eq_smul]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem LieSubalgebra.toEnd_eq (K : LieSubalgebra R L) {x : K} :
    LieModule.toEnd R K M x = LieModule.toEnd R L M x :=
  rfl


@[simp]
theorem LieSubalgebra.toEnd_mk (K : LieSubalgebra R L) {x : L} (hx : x ∈ K) :
    LieModule.toEnd R K M ⟨x, hx⟩ = LieModule.toEnd R L M x :=
  rfl


lemma LieSubmodule.coe_toEnd (N : LieSubmodule R L M) (x : L) (y : N) :
    (toEnd R L N x y : M) = toEnd R L M x y := rfl


lemma LieSubmodule.coe_toEnd_pow (N : LieSubmodule R L M) (x : L) (y : N) (n : ℕ) :
    ((toEnd R L N x ^ n) y : M) = (toEnd R L M x ^ n) y := by
  induction n generalizing y with
  | zero => rfl
  | succ n ih => simp only [pow_succ', LinearMap.mul_apply, ih, LieSubmodule.coe_toEnd]


lemma LieSubalgebra.coe_ad (H : LieSubalgebra R L) (x y : H) :
    (ad R H x y : L) = ad R L x y := rfl


lemma LieSubalgebra.coe_ad_pow (H : LieSubalgebra R L) (x y : H) (n : ℕ) :
    ((ad R H x ^ n) y : L) = (ad R L x ^ n) y :=
  LieSubmodule.coe_toEnd_pow R H L H.toLieSubmodule x y n


local notation "φ" => LieModule.toEnd R L M


lemma LieModule.toEnd_lie (x y : L) (z : M) :
    (φ x) ⁅y, z⁆ = ⁅ad R L x y, z⁆ + ⁅y, φ x z⁆ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x y : L
    z : M
    ⊢ Eq (((LieModule.toEnd R L M) x) (Bracket.bracket y z)) (HAdd.hAdd (Bracket.b …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma LieAlgebra.ad_lie (x y z : L) :
    (ad R L x) ⁅y, z⁆ = ⁅ad R L x y, z⁆ + ⁅y, ad R L x z⁆ :=
  toEnd_lie _ x y z


open Finset in
lemma LieModule.toEnd_pow_lie (x y : L) (z : M) (n : ℕ) :
    ((φ x) ^ n) ⁅y, z⁆ =
      ∑ ij ∈ antidiagonal n, n.choose ij.1 • ⁅((ad R L x) ^ ij.1) y, ((φ x) ^ ij.2) z⁆ := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_antidiagonal_choose_succ_nsmul
      (fun i j ↦ ⁅((ad R L x) ^ i) y, ((φ x) ^ j) z⁆) n]
    simp only [pow_succ', LinearMap.mul_apply, ih, map_sum, map_nsmul,
      toEnd_lie, nsmul_add, sum_add_distrib]
    rw [add_comm, add_left_cancel_iff, sum_congr rfl]
    rintro ⟨i, j⟩ hij
    rw [mem_antidiagonal] at hij
    rw [Nat.choose_symm_of_eq_add hij.symm]


open Finset in
lemma LieAlgebra.ad_pow_lie (x y z : L) (n : ℕ) :
    ((ad R L x) ^ n) ⁅y, z⁆ =
      ∑ ij ∈ antidiagonal n, n.choose ij.1 • ⁅((ad R L x) ^ ij.1) y, ((ad R L x) ^ ij.2) z⁆ :=
  toEnd_pow_lie _ x y z n


lemma toEnd_pow_comp_lieHom :
    (toEnd R L M₂ x ^ k) ∘ₗ f = f ∘ₗ toEnd R L M x ^ k := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    M₂ : Type w₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    k : Nat
    x : L
    ⊢ Eq (LinearMap.comp (HPow.hPow ((LieModule.toEnd R L M₂) x) k) ↑f) ((↑f).comp …
  -/
  apply LinearMap.commute_pow_left_of_commute
  /-
    case h
    R : Type u
    L : Type v
    M : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    M₂ : Type w₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    k : Nat
    x : L
    ⊢ Eq (LinearMap.comp ((LieModule.toEnd R L M₂) x) ↑f) ((↑f).comp ((LieModule.t …
  -/
  ext
  /-
    case h.h
    R : Type u
    L : Type v
    M : Type w
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    M₂ : Type w₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module R M₂
    inst✝¹ : LieRingModule L M₂
    inst✝ : LieModule R L M₂
    f : LieModuleHom R L M M₂
    k : Nat
    x : L
    x✝ : M
    ⊢ Eq ((LinearMap.comp ((LieModule.toEnd R L M₂) x) ↑f) x✝) (((↑f).comp ((LieMo …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma toEnd_pow_apply_map (m : M) :
    (toEnd R L M₂ x ^ k) (f m) = f ((toEnd R L M x ^ k) m) :=
  LinearMap.congr_fun (toEnd_pow_comp_lieHom f k x) m


theorem coe_map_toEnd_le :
    (N : Submodule R M).map (LieModule.toEnd R L M x) ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    x : L
    ⊢ LE.le (Submodule.map ((LieModule.toEnd R L M) x) ↑N) ↑N
  -/
  rintro n ⟨m, hm, rfl⟩
  /-
    case intro.intro
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    x : L
    m : M
    hm : Membership.mem (↑↑N) m
    ⊢ Membership.mem (↑N) (((LieModule.toEnd R L M) x) m)
  -/
  exact N.lie_mem hm
  /-
    🎉 no goals
  -/


theorem toEnd_comp_subtype_mem (m : M) (hm : m ∈ (N : Submodule R M)) :
    (toEnd R L M x).comp (N : Submodule R M).subtype ⟨m, hm⟩ ∈ (N : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    x : L
    m : M
    hm : Membership.mem (↑N) m
    ⊢ Membership.mem (↑N) ((LinearMap.comp ((LieModule.toEnd R L M) x) (↑N).subtyp …
  -/
  simpa using N.lie_mem hm
  /-
    🎉 no goals
  -/


@[simp]
theorem toEnd_restrict_eq_toEnd (h := N.toEnd_comp_subtype_mem x) :
    (toEnd R L M x).restrict h = toEnd R L N x := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    x : L
    h : optParam (∀ (m : M) (hm : Membership.mem (↑N) m), Membership.mem (↑N) ((Li …
    ⊢ Eq (LinearMap.restrict ((LieModule.toEnd R L M) x) h) ((LieModule.toEnd R L  …
  -/
  ext
  simp only [LinearMap.restrict_coe_apply, toEnd_apply_apply, ← coe_bracket,
    SetLike.coe_eq_coe]
  /-
    case h.a
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    x : L
    h : optParam (∀ (m : M) (hm : Membership.mem (↑N) m), Membership.mem (↑N) ((Li …
    x✝ : Subtype fun x => Membership.mem (↑N) x
    ⊢ Eq (Bracket.bracket x x✝) (((LieModule.toEnd R L (Subtype fun x => Membershi …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma mapsTo_pow_toEnd_sub_algebraMap {φ : R} {k : ℕ} {x : L} :
    MapsTo ((toEnd R L M x - algebraMap R (Module.End R M) φ) ^ k) N N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    φ : R
    k : Nat
    x : L
    ⊢ Set.MapsTo ⇑(HPow.hPow (HSub.hSub ((LieModule.toEnd R L M) x) ((algebraMap R …
  -/
  rw [LinearMap.coe_pow]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    φ : R
    k : Nat
    x : L
    ⊢ Set.MapsTo (Nat.iterate (⇑(HSub.hSub ((LieModule.toEnd R L M) x) ((algebraMa …
  -/
  exact MapsTo.iterate (fun m hm ↦ N.sub_mem (N.lie_mem hm) (N.smul_mem _ hm)) k
  /-
    🎉 no goals
  -/


theorem LieAlgebra.ad_eq_lmul_left_sub_lmul_right (A : Type v) [Ring A] [Algebra R A] :
    (ad R A : A → Module.End R A) = LinearMap.mulLeft R - LinearMap.mulRight R := by
  /-
    R : Type u
    inst✝² : CommRing R
    A : Type v
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    ⊢ Eq (⇑(LieAlgebra.ad R A)) (HSub.hSub (LinearMap.mulLeft R) (LinearMap.mulRig …
  -/
  ext a b; simp [LieRing.of_associative_ring_bracket]
           /-
             🎉 no goals
           -/


theorem LieSubalgebra.ad_comp_incl_eq (K : LieSubalgebra R L) (x : K) :
    (ad R L ↑x).comp (K.incl : K →ₗ[R] L) = (K.incl : K →ₗ[R] L).comp (ad R K x) := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    x : Subtype fun x => Membership.mem K x
    ⊢ Eq (LinearMap.comp ((LieAlgebra.ad R L) ↑x) ↑K.incl) ((↑K.incl).comp ((LieAl …
  -/
  ext y
  simp only [ad_apply, LieHom.coe_toLinearMap, LieSubalgebra.coe_incl, LinearMap.coe_comp,
    LieSubalgebra.coe_bracket, Function.comp_apply]


/-- A subalgebra of an associative algebra is a Lie subalgebra of the associated Lie algebra. -/
def lieSubalgebraOfSubalgebra (R : Type u) [CommRing R] (A : Type v) [Ring A] [Algebra R A]
    (A' : Subalgebra R A) : LieSubalgebra R A :=
  { Subalgebra.toSubmodule A' with
    lie_mem' := fun {x y} hx hy => by
      /-
        R : Type u
        inst✝² : CommRing R
        A : Type v
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        A' : Subalgebra R A
        x y : A
        hx : Membership.mem __src✝.carrier x
        hy : Membership.mem __src✝.carrier y
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x y)
      -/
      change ⁅x, y⁆ ∈ A'; change x ∈ A' at hx; change y ∈ A' at hy
      /-
        R : Type u
        inst✝² : CommRing R
        A : Type v
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        A' : Subalgebra R A
        x y : A
        hx : Membership.mem A' x
        hy : Membership.mem A' y
        ⊢ Membership.mem A' (Bracket.bracket x y)
      -/
      rw [LieRing.of_associative_ring_bracket]
      /-
        R : Type u
        inst✝² : CommRing R
        A : Type v
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        A' : Subalgebra R A
        x y : A
        hx : Membership.mem A' x
        hy : Membership.mem A' y
        ⊢ Membership.mem A' (HSub.hSub (HMul.hMul x y) (HMul.hMul y x))
      -/
      have hxy := A'.mul_mem hx hy
      /-
        R : Type u
        inst✝² : CommRing R
        A : Type v
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        A' : Subalgebra R A
        x y : A
        hx : Membership.mem A' x
        hy : Membership.mem A' y
        hxy : Membership.mem A' (HMul.hMul x y)
        ⊢ Membership.mem A' (HSub.hSub (HMul.hMul x y) (HMul.hMul y x))
      -/
      have hyx := A'.mul_mem hy hx
      /-
        R : Type u
        inst✝² : CommRing R
        A : Type v
        inst✝¹ : Ring A
        inst✝ : Algebra R A
        A' : Subalgebra R A
        x y : A
        hx : Membership.mem A' x
        hy : Membership.mem A' y
        hxy : Membership.mem A' (HMul.hMul x y)
        hyx : Membership.mem A' (HMul.hMul y x)
        ⊢ Membership.mem A' (HSub.hSub (HMul.hMul x y) (HMul.hMul y x))
      -/
      exact Submodule.sub_mem (Subalgebra.toSubmodule A') hxy hyx }
      /-
        🎉 no goals
      -/


/-- A linear equivalence of two modules induces a Lie algebra equivalence of their endomorphisms. -/
def lieConj : Module.End R M₁ ≃ₗ⁅R⁆ Module.End R M₂ :=
  { e.conj with
    map_lie' := fun {f g} =>
      show e.conj ⁅f, g⁆ = ⁅e.conj f, e.conj g⁆ by
        simp only [LieRing.of_associative_ring_bracket, LinearMap.mul_eq_comp, e.conj_comp,
          map_sub] }


@[simp]
theorem lieConj_apply (f : Module.End R M₁) : e.lieConj f = e.conj f :=
  rfl


@[simp]
theorem lieConj_symm : e.lieConj.symm = e.symm.lieConj :=
  rfl


/-- An equivalence of associative algebras is an equivalence of associated Lie algebras. -/
def toLieEquiv : A₁ ≃ₗ⁅R⁆ A₂ :=
  { e.toLinearEquiv with
    toFun := e.toFun
    map_lie' := fun {x y} => by
      /-
        R : Type u
        A₁ : Type v
        A₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : Ring A₁
        inst✝² : Ring A₂
        inst✝¹ : Algebra R A₁
        inst✝ : Algebra R A₂
        e : AlgEquiv R A₁ A₂
        x y : A₁
        ⊢ Eq ({ toFun := e.toFun, map_add' := ⋯, map_smul' := ⋯ }.toFun (Bracket.brack …
      -/
      have : e.toEquiv.toFun = e := rfl
      /-
        R : Type u
        A₁ : Type v
        A₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : Ring A₁
        inst✝² : Ring A₂
        inst✝¹ : Algebra R A₁
        inst✝ : Algebra R A₂
        e : AlgEquiv R A₁ A₂
        x y : A₁
        this : Eq e.toFun ⇑e
        ⊢ Eq ({ toFun := e.toFun, map_add' := ⋯, map_smul' := ⋯ }.toFun (Bracket.brack …
      -/
      simp_rw [LieRing.of_associative_ring_bracket, this, map_sub, map_mul] }
      /-
        🎉 no goals
      -/


@[simp]
theorem toLieEquiv_apply (x : A₁) : e.toLieEquiv x = e x :=
  rfl


@[simp]
theorem toLieEquiv_symm_apply (x : A₂) : e.toLieEquiv.symm x = e.symm x :=
  rfl


/-- Given an equivalence `e` of Lie algebras from `L` to `L'`, and an element `x : L`, the conjugate
of the endomorphism `ad(x)` of `L` by `e` is the endomorphism `ad(e x)` of `L'`. -/
@[simp]
lemma conj_ad_apply (e : L ≃ₗ⁅R⁆ L') (x : L) : LinearEquiv.conj e (ad R L x) = ad R L' (e x) := by
  /-
    R : Type u_1
    L : Type u_2
    L' : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : LieRing L'
    inst✝ : LieAlgebra R L'
    e : LieEquiv R L L'
    x : L
    ⊢ Eq (e.toLinearEquiv.conj ((LieAlgebra.ad R L) x)) ((LieAlgebra.ad R L') (e x))
  -/
  ext y'
  rw [LinearEquiv.conj_apply_apply, ad_apply, ad_apply, coe_toLinearEquiv, map_lie,
    ← coe_toLinearEquiv, LinearEquiv.apply_symm_apply]


