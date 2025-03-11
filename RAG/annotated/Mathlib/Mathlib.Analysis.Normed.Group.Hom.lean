/-- A morphism of seminormed abelian groups is a bounded group homomorphism. -/
structure NormedAddGroupHom (V W : Type*) [SeminormedAddCommGroup V]
  [SeminormedAddCommGroup W] where
  /-- The function underlying a `NormedAddGroupHom` -/
  toFun : V → W
  /-- A `NormedAddGroupHom` is additive. -/
  map_add' : ∀ v₁ v₂, toFun (v₁ + v₂) = toFun v₁ + toFun v₂
  /-- A `NormedAddGroupHom` is bounded. -/
  bound' : ∃ C, ∀ v, ‖toFun v‖ ≤ C * ‖v‖


/-- Associate to a group homomorphism a bounded group homomorphism under a norm control condition.

See `AddMonoidHom.mkNormedAddGroupHom'` for a version that uses `ℝ≥0` for the bound. -/
def mkNormedAddGroupHom (f : V →+ W) (C : ℝ) (h : ∀ v, ‖f v‖ ≤ C * ‖v‖) : NormedAddGroupHom V W :=
  { f with bound' := ⟨C, h⟩ }


/-- Associate to a group homomorphism a bounded group homomorphism under a norm control condition.

See `AddMonoidHom.mkNormedAddGroupHom` for a version that uses `ℝ` for the bound. -/
def mkNormedAddGroupHom' (f : V →+ W) (C : ℝ≥0) (hC : ∀ x, ‖f x‖₊ ≤ C * ‖x‖₊) :
    NormedAddGroupHom V W :=
  { f with bound' := ⟨C, hC⟩ }


theorem exists_pos_bound_of_bound {V W : Type*} [SeminormedAddCommGroup V]
    [SeminormedAddCommGroup W] {f : V → W} (M : ℝ) (h : ∀ x, ‖f x‖ ≤ M * ‖x‖) :
    ∃ N, 0 < N ∧ ∀ x, ‖f x‖ ≤ N * ‖x‖ :=
  ⟨max M 1, lt_of_lt_of_le zero_lt_one (le_max_right _ _), fun x =>
    calc
      ‖f x‖ ≤ M * ‖x‖ := h x
                              /-
                                V : Type u_1
                                W : Type u_2
                                inst✝¹ : SeminormedAddCommGroup V
                                inst✝ : SeminormedAddCommGroup W
                                f : V → W
                                M : Real
                                h : ∀ (x : V), LE.le (Norm.norm (f x)) (HMul.hMul M (Norm.norm x))
                                x : V
                                ⊢ LE.le (HMul.hMul M (Norm.norm x)) (HMul.hMul (Max.max M 1) (Norm.norm x))
                              -/
      _ ≤ max M 1 * ‖x‖ := by gcongr; apply le_max_left
                                      /-
                                        🎉 no goals
                                      -/
      ⟩


/-- A Lipschitz continuous additive homomorphism is a normed additive group homomorphism. -/
def ofLipschitz (f : V₁ →+ V₂) {K : ℝ≥0} (h : LipschitzWith K f) : NormedAddGroupHom V₁ V₂ :=
                                     /-
                                       V : Type u_1
                                       V₁ : Type u_2
                                       V₂ : Type u_3
                                       V₃ : Type u_4
                                       inst✝³ : SeminormedAddCommGroup V
                                       inst✝² : SeminormedAddCommGroup V₁
                                       inst✝¹ : SeminormedAddCommGroup V₂
                                       inst✝ : SeminormedAddCommGroup V₃
                                       f✝ g : NormedAddGroupHom V₁ V₂
                                       f : AddMonoidHom V₁ V₂
                                       K : NNReal
                                       h : LipschitzWith K ⇑f
                                       x : V₁
                                       ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (↑K) (Norm.norm x))
                                     -/
  f.mkNormedAddGroupHom K fun x ↦ by simpa only [map_zero, dist_zero_right] using h.dist_le_mul x 0
                                     /-
                                       🎉 no goals
                                     -/


instance funLike : FunLike (NormedAddGroupHom V₁ V₂) V₁ V₂ where
  coe := toFun
                             /-
                               V : Type u_1
                               V₁ : Type u_2
                               V₂ : Type u_3
                               V₃ : Type u_4
                               inst✝³ : SeminormedAddCommGroup V
                               inst✝² : SeminormedAddCommGroup V₁
                               inst✝¹ : SeminormedAddCommGroup V₂
                               inst✝ : SeminormedAddCommGroup V₃
                               f✝ g✝ f g : NormedAddGroupHom V₁ V₂
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/

-- Porting note: moved this declaration up so we could get a `FunLike` instance sooner.

instance toAddMonoidHomClass : AddMonoidHomClass (NormedAddGroupHom V₁ V₂) V₁ V₂ where
  map_add f := f.map_add'
  map_zero f := (AddMonoidHom.mk' f.toFun f.map_add').map_zero


theorem coe_inj (H : (f : V₁ → V₂) = g) : f = g := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f g : NormedAddGroupHom V₁ V₂
    H : Eq ⇑f ⇑g
    ⊢ Eq f g
  -/
  cases f; cases g; congr
                    /-
                      🎉 no goals
                    -/


theorem coe_injective : @Function.Injective (NormedAddGroupHom V₁ V₂) (V₁ → V₂) toFun := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    ⊢ Function.Injective NormedAddGroupHom.toFun
  -/
  apply coe_inj
  /-
    🎉 no goals
  -/


theorem coe_inj_iff : f = g ↔ (f : V₁ → V₂) = g :=
  ⟨congr_arg _, coe_inj⟩


@[ext]
theorem ext (H : ∀ x, f x = g x) : f = g :=
  coe_inj <| funext H


@[simp]
theorem toFun_eq_coe : f.toFun = f :=
  rfl

-- Porting note: removed `simp` because `simpNF` complains the LHS doesn't simplify.

theorem coe_mk (f) (h₁) (h₂) (h₃) : ⇑(⟨f, h₁, h₂, h₃⟩ : NormedAddGroupHom V₁ V₂) = f :=
  rfl


@[simp]
theorem coe_mkNormedAddGroupHom (f : V₁ →+ V₂) (C) (hC) : ⇑(f.mkNormedAddGroupHom C hC) = f :=
  rfl


@[simp]
theorem coe_mkNormedAddGroupHom' (f : V₁ →+ V₂) (C) (hC) : ⇑(f.mkNormedAddGroupHom' C hC) = f :=
  rfl


/-- The group homomorphism underlying a bounded group homomorphism. -/
def toAddMonoidHom (f : NormedAddGroupHom V₁ V₂) : V₁ →+ V₂ :=
  AddMonoidHom.mk' f f.map_add'


@[simp]
theorem coe_toAddMonoidHom : ⇑f.toAddMonoidHom = f :=
  rfl


theorem toAddMonoidHom_injective :
    Function.Injective (@NormedAddGroupHom.toAddMonoidHom V₁ V₂ _ _) := fun f g h =>
                /-
                  V₁ : Type u_2
                  V₂ : Type u_3
                  inst✝¹ : SeminormedAddCommGroup V₁
                  inst✝ : SeminormedAddCommGroup V₂
                  f g : NormedAddGroupHom V₁ V₂
                  h : Eq f.toAddMonoidHom g.toAddMonoidHom
                  ⊢ Eq ⇑f ⇑g
                -/
  coe_inj <| by rw [← coe_toAddMonoidHom f, ← coe_toAddMonoidHom g, h]
                /-
                  🎉 no goals
                -/


@[simp]
theorem mk_toAddMonoidHom (f) (h₁) (h₂) :
    (⟨f, h₁, h₂⟩ : NormedAddGroupHom V₁ V₂).toAddMonoidHom = AddMonoidHom.mk' f h₁ :=
  rfl


theorem bound : ∃ C, 0 < C ∧ ∀ x, ‖f x‖ ≤ C * ‖x‖ :=
  let ⟨_C, hC⟩ := f.bound'
  exists_pos_bound_of_bound _ hC


theorem antilipschitz_of_norm_ge {K : ℝ≥0} (h : ∀ x, ‖x‖ ≤ K * ‖f x‖) : AntilipschitzWith K f :=
                                                 /-
                                                   V₁ : Type u_2
                                                   V₂ : Type u_3
                                                   inst✝¹ : SeminormedAddCommGroup V₁
                                                   inst✝ : SeminormedAddCommGroup V₂
                                                   f : NormedAddGroupHom V₁ V₂
                                                   K : NNReal
                                                   h : ∀ (x : V₁), LE.le (Norm.norm x) (HMul.hMul (↑K) (Norm.norm (f x)))
                                                   x y : V₁
                                                   ⊢ LE.le (Dist.dist x y) (HMul.hMul (↑K) (Dist.dist (f x) (f y)))
                                                 -/
  AntilipschitzWith.of_le_mul_dist fun x y => by simpa only [dist_eq_norm, map_sub] using h (x - y)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A normed group hom is surjective on the subgroup `K` with constant `C` if every element
`x` of `K` has a preimage whose norm is bounded above by `C*‖x‖`. This is a more
abstract version of `f` having a right inverse defined on `K` with operator norm
at most `C`. -/
def SurjectiveOnWith (f : NormedAddGroupHom V₁ V₂) (K : AddSubgroup V₂) (C : ℝ) : Prop :=
  ∀ h ∈ K, ∃ g, f g = h ∧ ‖g‖ ≤ C * ‖h‖


theorem SurjectiveOnWith.mono {f : NormedAddGroupHom V₁ V₂} {K : AddSubgroup V₂} {C C' : ℝ}
    (h : f.SurjectiveOnWith K C) (H : C ≤ C') : f.SurjectiveOnWith K C' := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    K : AddSubgroup V₂
    C C' : Real
    h : f.SurjectiveOnWith K C
    H : LE.le C C'
    ⊢ f.SurjectiveOnWith K C'
  -/
  intro k k_in
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    K : AddSubgroup V₂
    C C' : Real
    h : f.SurjectiveOnWith K C
    H : LE.le C C'
    k : V₂
    k_in : Membership.mem K k
    ⊢ Exists fun g => And (Eq (f g) k) (LE.le (Norm.norm g) (HMul.hMul C' (Norm.no …
  -/
  rcases h k k_in with ⟨g, rfl, hg⟩
  /-
    case intro.intro
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    K : AddSubgroup V₂
    C C' : Real
    h : f.SurjectiveOnWith K C
    H : LE.le C C'
    g : V₁
    k_in : Membership.mem K (f g)
    hg : LE.le (Norm.norm g) (HMul.hMul C (Norm.norm (f g)))
    ⊢ Exists fun g_1 => And (Eq (f g_1) (f g)) (LE.le (Norm.norm g_1) (HMul.hMul C …
  -/
  use g, rfl
  /-
    case right
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    K : AddSubgroup V₂
    C C' : Real
    h : f.SurjectiveOnWith K C
    H : LE.le C C'
    g : V₁
    k_in : Membership.mem K (f g)
    hg : LE.le (Norm.norm g) (HMul.hMul C (Norm.norm (f g)))
    ⊢ LE.le (Norm.norm g) (HMul.hMul C' (Norm.norm (f g)))
  -/
  by_cases Hg : ‖f g‖ = 0
    /-
      case pos
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      K : AddSubgroup V₂
      C C' : Real
      h : f.SurjectiveOnWith K C
      H : LE.le C C'
      g : V₁
      k_in : Membership.mem K (f g)
      hg : LE.le (Norm.norm g) (HMul.hMul C (Norm.norm (f g)))
      Hg : Eq (Norm.norm (f g)) 0
      ⊢ LE.le (Norm.norm g) (HMul.hMul C' (Norm.norm (f g)))
    -/
  · simpa [Hg] using hg
    /-
      🎉 no goals
    -/
    /-
      case neg
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      K : AddSubgroup V₂
      C C' : Real
      h : f.SurjectiveOnWith K C
      H : LE.le C C'
      g : V₁
      k_in : Membership.mem K (f g)
      hg : LE.le (Norm.norm g) (HMul.hMul C (Norm.norm (f g)))
      Hg : Not (Eq (Norm.norm (f g)) 0)
      ⊢ LE.le (Norm.norm g) (HMul.hMul C' (Norm.norm (f g)))
    -/
  · exact hg.trans (by gcongr)
    /-
      🎉 no goals
    -/


theorem SurjectiveOnWith.exists_pos {f : NormedAddGroupHom V₁ V₂} {K : AddSubgroup V₂} {C : ℝ}
    (h : f.SurjectiveOnWith K C) : ∃ C' > 0, f.SurjectiveOnWith K C' := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    K : AddSubgroup V₂
    C : Real
    h : f.SurjectiveOnWith K C
    ⊢ Exists fun C' => And (GT.gt C' 0) (f.SurjectiveOnWith K C')
  -/
  refine ⟨|C| + 1, ?_, ?_⟩
    /-
      case refine_1
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      K : AddSubgroup V₂
      C : Real
      h : f.SurjectiveOnWith K C
      ⊢ GT.gt (HAdd.hAdd (abs C) 1) 0
    -/
  · linarith [abs_nonneg C]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      K : AddSubgroup V₂
      C : Real
      h : f.SurjectiveOnWith K C
      ⊢ f.SurjectiveOnWith K (HAdd.hAdd (abs C) 1)
    -/
  · apply h.mono
    /-
      case refine_2
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      K : AddSubgroup V₂
      C : Real
      h : f.SurjectiveOnWith K C
      ⊢ LE.le C (HAdd.hAdd (abs C) 1)
    -/
    linarith [le_abs_self C]
    /-
      🎉 no goals
    -/


theorem SurjectiveOnWith.surjOn {f : NormedAddGroupHom V₁ V₂} {K : AddSubgroup V₂} {C : ℝ}
    (h : f.SurjectiveOnWith K C) : Set.SurjOn f Set.univ K := fun x hx =>
  (h x hx).imp fun _a ⟨ha, _⟩ => ⟨Set.mem_univ _, ha⟩


/-- The operator norm of a seminormed group homomorphism is the inf of all its bounds. -/
def opNorm (f : NormedAddGroupHom V₁ V₂) :=
  sInf { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ }


instance hasOpNorm : Norm (NormedAddGroupHom V₁ V₂) :=
  ⟨opNorm⟩


theorem norm_def : ‖f‖ = sInf { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ } :=
  rfl

-- So that invocations of `le_csInf` make sense: we show that the set of
-- bounds is nonempty and bounded below.

theorem bounds_nonempty {f : NormedAddGroupHom V₁ V₂} :
    ∃ c, c ∈ { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ } :=
  let ⟨M, hMp, hMb⟩ := f.bound
  ⟨M, le_of_lt hMp, hMb⟩


theorem bounds_bddBelow {f : NormedAddGroupHom V₁ V₂} :
    BddBelow { c | 0 ≤ c ∧ ∀ x, ‖f x‖ ≤ c * ‖x‖ } :=
  ⟨0, fun _ ⟨hn, _⟩ => hn⟩


theorem opNorm_nonneg : 0 ≤ ‖f‖ :=
  le_csInf bounds_nonempty fun _ ⟨hx, _⟩ => hx


/-- The fundamental property of the operator norm: `‖f x‖ ≤ ‖f‖ * ‖x‖`. -/
theorem le_opNorm (x : V₁) : ‖f x‖ ≤ ‖f‖ * ‖x‖ := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    x : V₁
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (Norm.norm x))
  -/
  obtain ⟨C, _Cpos, hC⟩ := f.bound
  /-
    case intro.intro
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    x : V₁
    C : Real
    _Cpos : LT.lt 0 C
    hC : ∀ (x : V₁), LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (Norm.norm x))
  -/
  replace hC := hC x
  /-
    case intro.intro
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    x : V₁
    C : Real
    _Cpos : LT.lt 0 C
    hC : LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (Norm.norm x))
  -/
  by_cases h : ‖x‖ = 0
    /-
      case pos
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      x : V₁
      C : Real
      _Cpos : LT.lt 0 C
      hC : LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
      h : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (Norm.norm x))
    -/
  · rwa [h, mul_zero] at hC ⊢
    /-
      🎉 no goals
    -/
  /-
    case neg
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    x : V₁
    C : Real
    _Cpos : LT.lt 0 C
    hC : LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
    h : Not (Eq (Norm.norm x) 0)
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm f) (Norm.norm x))
  -/
  have hlt : 0 < ‖x‖ := lt_of_le_of_ne (norm_nonneg x) (Ne.symm h)
  exact
    (div_le_iff₀ hlt).mp
      (le_csInf bounds_nonempty fun c ⟨_, hc⟩ => (div_le_iff₀ hlt).mpr <| by apply hc)


theorem le_opNorm_of_le {c : ℝ} {x} (h : ‖x‖ ≤ c) : ‖f x‖ ≤ ‖f‖ * c :=
                               /-
                                 V₁ : Type u_2
                                 V₂ : Type u_3
                                 inst✝¹ : SeminormedAddCommGroup V₁
                                 inst✝ : SeminormedAddCommGroup V₂
                                 f : NormedAddGroupHom V₁ V₂
                                 c : Real
                                 x : V₁
                                 h : LE.le (Norm.norm x) c
                                 ⊢ LE.le (HMul.hMul (Norm.norm f) (Norm.norm x)) (HMul.hMul (Norm.norm f) c)
                               -/
  le_trans (f.le_opNorm x) (by gcongr; exact f.opNorm_nonneg)
                                       /-
                                         🎉 no goals
                                       -/


theorem le_of_opNorm_le {c : ℝ} (h : ‖f‖ ≤ c) (x : V₁) : ‖f x‖ ≤ c * ‖x‖ :=
                            /-
                              V₁ : Type u_2
                              V₂ : Type u_3
                              inst✝¹ : SeminormedAddCommGroup V₁
                              inst✝ : SeminormedAddCommGroup V₂
                              f : NormedAddGroupHom V₁ V₂
                              c : Real
                              h : LE.le (Norm.norm f) c
                              x : V₁
                              ⊢ LE.le (HMul.hMul (Norm.norm f) (Norm.norm x)) (HMul.hMul c (Norm.norm x))
                            -/
  (f.le_opNorm x).trans (by gcongr)
                            /-
                              🎉 no goals
                            -/


/-- continuous linear maps are Lipschitz continuous. -/
theorem lipschitz : LipschitzWith ⟨‖f‖, opNorm_nonneg f⟩ f :=
  LipschitzWith.of_dist_le_mul fun x y => by
    /-
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      x y : V₁
      ⊢ LE.le (Dist.dist (f x) (f y)) (HMul.hMul (↑⟨Norm.norm f, ⋯⟩) (Dist.dist x y))
    -/
    rw [dist_eq_norm, dist_eq_norm, ← map_sub]
    /-
      V₁ : Type u_2
      V₂ : Type u_3
      inst✝¹ : SeminormedAddCommGroup V₁
      inst✝ : SeminormedAddCommGroup V₂
      f : NormedAddGroupHom V₁ V₂
      x y : V₁
      ⊢ LE.le (Norm.norm (f (HSub.hSub x y))) (HMul.hMul (↑⟨Norm.norm f, ⋯⟩) (Norm.n …
    -/
    apply le_opNorm
    /-
      🎉 no goals
    -/


protected theorem uniformContinuous (f : NormedAddGroupHom V₁ V₂) : UniformContinuous f :=
  f.lipschitz.uniformContinuous


@[continuity]
protected theorem continuous (f : NormedAddGroupHom V₁ V₂) : Continuous f :=
  f.uniformContinuous.continuous


theorem ratio_le_opNorm (x : V₁) : ‖f x‖ / ‖x‖ ≤ ‖f‖ :=
  div_le_of_le_mul₀ (norm_nonneg _) f.opNorm_nonneg (le_opNorm _ _)


/-- If one controls the norm of every `f x`, then one controls the norm of `f`. -/
theorem opNorm_le_bound {M : ℝ} (hMp : 0 ≤ M) (hM : ∀ x, ‖f x‖ ≤ M * ‖x‖) : ‖f‖ ≤ M :=
  csInf_le bounds_bddBelow ⟨hMp, hM⟩


theorem opNorm_eq_of_bounds {M : ℝ} (M_nonneg : 0 ≤ M) (h_above : ∀ x, ‖f x‖ ≤ M * ‖x‖)
    (h_below : ∀ N ≥ 0, (∀ x, ‖f x‖ ≤ N * ‖x‖) → M ≤ N) : ‖f‖ = M :=
  le_antisymm (f.opNorm_le_bound M_nonneg h_above)
    ((le_csInf_iff NormedAddGroupHom.bounds_bddBelow ⟨M, M_nonneg, h_above⟩).mpr
      fun N ⟨N_nonneg, hN⟩ => h_below N N_nonneg hN)


theorem opNorm_le_of_lipschitz {f : NormedAddGroupHom V₁ V₂} {K : ℝ≥0} (hf : LipschitzWith K f) :
    ‖f‖ ≤ K :=
                                    /-
                                      V₁ : Type u_2
                                      V₂ : Type u_3
                                      inst✝¹ : SeminormedAddCommGroup V₁
                                      inst✝ : SeminormedAddCommGroup V₂
                                      f : NormedAddGroupHom V₁ V₂
                                      K : NNReal
                                      hf : LipschitzWith K ⇑f
                                      x : V₁
                                      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (↑K) (Norm.norm x))
                                    -/
  f.opNorm_le_bound K.2 fun x => by simpa only [dist_zero_right, map_zero] using hf.dist_le_mul x 0
                                    /-
                                      🎉 no goals
                                    -/


/-- If a bounded group homomorphism map is constructed from a group homomorphism via the constructor
`AddMonoidHom.mkNormedAddGroupHom`, then its norm is bounded by the bound given to the constructor
if it is nonnegative. -/
theorem mkNormedAddGroupHom_norm_le (f : V₁ →+ V₂) {C : ℝ} (hC : 0 ≤ C) (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    ‖f.mkNormedAddGroupHom C h‖ ≤ C :=
  opNorm_le_bound _ hC h


/-- If a bounded group homomorphism map is constructed from a group homomorphism via the constructor
`NormedAddGroupHom.ofLipschitz`, then its norm is bounded by the bound given to the constructor. -/
theorem ofLipschitz_norm_le (f : V₁ →+ V₂) {K : ℝ≥0} (h : LipschitzWith K f) :
    ‖ofLipschitz f h‖ ≤ K :=
  mkNormedAddGroupHom_norm_le f K.coe_nonneg _


/-- If a bounded group homomorphism map is constructed from a group homomorphism
via the constructor `AddMonoidHom.mkNormedAddGroupHom`, then its norm is bounded by the bound
given to the constructor or zero if this bound is negative. -/
theorem mkNormedAddGroupHom_norm_le' (f : V₁ →+ V₂) {C : ℝ} (h : ∀ x, ‖f x‖ ≤ C * ‖x‖) :
    ‖f.mkNormedAddGroupHom C h‖ ≤ max C 0 :=
  opNorm_le_bound _ (le_max_right _ _) fun x =>
                      /-
                        V₁ : Type u_2
                        V₂ : Type u_3
                        inst✝¹ : SeminormedAddCommGroup V₁
                        inst✝ : SeminormedAddCommGroup V₂
                        f : AddMonoidHom V₁ V₂
                        C : Real
                        h : ∀ (x : V₁), LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm x))
                        x : V₁
                        ⊢ LE.le (HMul.hMul C (Norm.norm x)) (HMul.hMul (Max.max C 0) (Norm.norm x))
                      -/
    (h x).trans <| by gcongr; apply le_max_left
                              /-
                                🎉 no goals
                              -/


alias _root_.AddMonoidHom.mkNormedAddGroupHom_norm_le := mkNormedAddGroupHom_norm_le


alias _root_.AddMonoidHom.mkNormedAddGroupHom_norm_le' := mkNormedAddGroupHom_norm_le'


/-- Addition of normed group homs. -/
instance add : Add (NormedAddGroupHom V₁ V₂) :=
  ⟨fun f g =>
    (f.toAddMonoidHom + g.toAddMonoidHom).mkNormedAddGroupHom (‖f‖ + ‖g‖) fun v =>
      calc
        ‖f v + g v‖ ≤ ‖f v‖ + ‖g v‖ := norm_add_le _ _
                                        /-
                                          V : Type u_1
                                          V₁ : Type u_2
                                          V₂ : Type u_3
                                          V₃ : Type u_4
                                          inst✝³ : SeminormedAddCommGroup V
                                          inst✝² : SeminormedAddCommGroup V₁
                                          inst✝¹ : SeminormedAddCommGroup V₂
                                          inst✝ : SeminormedAddCommGroup V₃
                                          f✝ g✝ f g : NormedAddGroupHom V₁ V₂
                                          v : V₁
                                          ⊢ LE.le (HAdd.hAdd (Norm.norm (f v)) (Norm.norm (g v))) (HAdd.hAdd (HMul.hMul  …
                                        -/
                                                   /-
                                                     🎉 no goals
                                                   -/
        _ ≤ ‖f‖ * ‖v‖ + ‖g‖ * ‖v‖ := by gcongr <;> apply le_opNorm
                                                   /-
                                                     🎉 no goals
                                                   -/
                                    /-
                                      V : Type u_1
                                      V₁ : Type u_2
                                      V₂ : Type u_3
                                      V₃ : Type u_4
                                      inst✝³ : SeminormedAddCommGroup V
                                      inst✝² : SeminormedAddCommGroup V₁
                                      inst✝¹ : SeminormedAddCommGroup V₂
                                      inst✝ : SeminormedAddCommGroup V₃
                                      f✝ g✝ f g : NormedAddGroupHom V₁ V₂
                                      v : V₁
                                      ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm f) (Norm.norm v)) (HMul.hMul (Norm.norm  …
                                    -/
        _ = (‖f‖ + ‖g‖) * ‖v‖ := by rw [add_mul]
                                    /-
                                      🎉 no goals
                                    -/
        ⟩


/-- The operator norm satisfies the triangle inequality. -/
theorem opNorm_add_le : ‖f + g‖ ≤ ‖f‖ + ‖g‖ :=
  mkNormedAddGroupHom_norm_le _ (add_nonneg (opNorm_nonneg _) (opNorm_nonneg _)) _

-- Porting note: this library note doesn't seem to apply anymore
/-
library_note "addition on function coercions"/--
Terms containing `@has_add.add (has_coe_to_fun.F ...) pi.has_add`
seem to cause leanchecker to [crash due to an out-of-memory
condition](https://github.com/leanprover-community/lean/issues/543).
As a workaround, we add a type annotation: `(f + g : V₁ → V₂)`
-/
-/


@[simp]
theorem coe_add (f g : NormedAddGroupHom V₁ V₂) : ⇑(f + g) = f + g :=
  rfl


@[simp]
theorem add_apply (f g : NormedAddGroupHom V₁ V₂) (v : V₁) :
    (f + g) v = f v + g v :=
  rfl


instance zero : Zero (NormedAddGroupHom V₁ V₂) :=
                                            /-
                                              V : Type u_1
                                              V₁ : Type u_2
                                              V₂ : Type u_3
                                              V₃ : Type u_4
                                              inst✝³ : SeminormedAddCommGroup V
                                              inst✝² : SeminormedAddCommGroup V₁
                                              inst✝¹ : SeminormedAddCommGroup V₂
                                              inst✝ : SeminormedAddCommGroup V₃
                                              f g : NormedAddGroupHom V₁ V₂
                                              ⊢ ∀ (v : V₁), LE.le (Norm.norm (0 v)) (HMul.hMul 0 (Norm.norm v))
                                            -/
  ⟨(0 : V₁ →+ V₂).mkNormedAddGroupHom 0 (by simp)⟩
                                            /-
                                              🎉 no goals
                                            -/


instance inhabited : Inhabited (NormedAddGroupHom V₁ V₂) :=
  ⟨0⟩


/-- The norm of the `0` operator is `0`. -/
theorem opNorm_zero : ‖(0 : NormedAddGroupHom V₁ V₂)‖ = 0 :=
  le_antisymm
    (csInf_le bounds_bddBelow
      ⟨ge_of_eq rfl, fun _ =>
        le_of_eq
          (by
            /-
              V₁ : Type u_2
              V₂ : Type u_3
              inst✝¹ : SeminormedAddCommGroup V₁
              inst✝ : SeminormedAddCommGroup V₂
              x✝ : V₁
              ⊢ Eq (Norm.norm (0 x✝)) (HMul.hMul 0 (Norm.norm x✝))
            -/
            rw [zero_mul]
            /-
              V₁ : Type u_2
              V₂ : Type u_3
              inst✝¹ : SeminormedAddCommGroup V₁
              inst✝ : SeminormedAddCommGroup V₂
              x✝ : V₁
              ⊢ Eq (Norm.norm (0 x✝)) 0
            -/
            exact norm_zero)⟩)
            /-
              🎉 no goals
            -/
    (opNorm_nonneg _)


/-- For normed groups, an operator is zero iff its norm vanishes. -/
theorem opNorm_zero_iff {V₁ V₂ : Type*} [NormedAddCommGroup V₁] [NormedAddCommGroup V₂]
    {f : NormedAddGroupHom V₁ V₂} : ‖f‖ = 0 ↔ f = 0 :=
  Iff.intro
    (fun hn =>
      ext fun x =>
        norm_le_zero_iff.1
          (calc
            _ ≤ ‖f‖ * ‖x‖ := le_opNorm _ _
                        /-
                          V₁ : Type u_5
                          V₂ : Type u_6
                          inst✝¹ : NormedAddCommGroup V₁
                          inst✝ : NormedAddCommGroup V₂
                          f : NormedAddGroupHom V₁ V₂
                          hn : Eq (Norm.norm f) 0
                          x : V₁
                          ⊢ Eq (HMul.hMul (Norm.norm f) (Norm.norm x)) 0
                        -/
            _ = _ := by rw [hn, zero_mul]
                        /-
                          🎉 no goals
                        -/
            ))
                 /-
                   V₁ : Type u_5
                   V₂ : Type u_6
                   inst✝¹ : NormedAddCommGroup V₁
                   inst✝ : NormedAddCommGroup V₂
                   f : NormedAddGroupHom V₁ V₂
                   hf : Eq f 0
                   ⊢ Eq (Norm.norm f) 0
                 -/
    fun hf => by rw [hf, opNorm_zero]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem coe_zero : ⇑(0 : NormedAddGroupHom V₁ V₂) = 0 :=
  rfl


@[simp]
theorem zero_apply (v : V₁) : (0 : NormedAddGroupHom V₁ V₂) v = 0 :=
  rfl


/-- The identity as a continuous normed group hom. -/
@[simps!]
def id : NormedAddGroupHom V V :=
                                                /-
                                                  V : Type u_1
                                                  V₁ : Type u_2
                                                  V₂ : Type u_3
                                                  V₃ : Type u_4
                                                  inst✝³ : SeminormedAddCommGroup V
                                                  inst✝² : SeminormedAddCommGroup V₁
                                                  inst✝¹ : SeminormedAddCommGroup V₂
                                                  inst✝ : SeminormedAddCommGroup V₃
                                                  f g : NormedAddGroupHom V₁ V₂
                                                  ⊢ ∀ (v : V), LE.le (Norm.norm ((AddMonoidHom.id V) v)) (HMul.hMul 1 (Norm.norm …
                                                -/
  (AddMonoidHom.id V).mkNormedAddGroupHom 1 (by simp [le_refl])
                                                /-
                                                  🎉 no goals
                                                -/


/-- The norm of the identity is at most `1`. It is in fact `1`, except when the norm of every
element vanishes, where it is `0`. (Since we are working with seminorms this can happen even if the
space is non-trivial.) It means that one can not do better than an inequality in general. -/
theorem norm_id_le : ‖(id V : NormedAddGroupHom V V)‖ ≤ 1 :=
                                            /-
                                              V : Type u_1
                                              inst✝ : SeminormedAddCommGroup V
                                              x : V
                                              ⊢ LE.le (Norm.norm ((NormedAddGroupHom.id V) x)) (HMul.hMul 1 (Norm.norm x))
                                            -/
  opNorm_le_bound _ zero_le_one fun x => by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- If there is an element with norm different from `0`, then the norm of the identity equals `1`.
(Since we are working with seminorms supposing that the space is non-trivial is not enough.) -/
theorem norm_id_of_nontrivial_seminorm (h : ∃ x : V, ‖x‖ ≠ 0) : ‖id V‖ = 1 :=
  le_antisymm (norm_id_le V) <| by
    /-
      V : Type u_1
      inst✝ : SeminormedAddCommGroup V
      h : Exists fun x => Ne (Norm.norm x) 0
      ⊢ LE.le 1 (Norm.norm (NormedAddGroupHom.id V))
    -/
    let ⟨x, hx⟩ := h
    /-
      V : Type u_1
      inst✝ : SeminormedAddCommGroup V
      h : Exists fun x => Ne (Norm.norm x) 0
      x : V
      hx : Ne (Norm.norm x) 0
      ⊢ LE.le 1 (Norm.norm (NormedAddGroupHom.id V))
    -/
    have := (id V).ratio_le_opNorm x
    /-
      V : Type u_1
      inst✝ : SeminormedAddCommGroup V
      h : Exists fun x => Ne (Norm.norm x) 0
      x : V
      hx : Ne (Norm.norm x) 0
      this : LE.le (HDiv.hDiv (Norm.norm ((NormedAddGroupHom.id V) x)) (Norm.norm x) …
      ⊢ LE.le 1 (Norm.norm (NormedAddGroupHom.id V))
    -/
    rwa [id_apply, div_self hx] at this
    /-
      🎉 no goals
    -/


/-- If a normed space is non-trivial, then the norm of the identity equals `1`. -/
theorem norm_id {V : Type*} [NormedAddCommGroup V] [Nontrivial V] : ‖id V‖ = 1 := by
  /-
    V : Type u_5
    inst✝¹ : NormedAddCommGroup V
    inst✝ : Nontrivial V
    ⊢ Eq (Norm.norm (NormedAddGroupHom.id V)) 1
  -/
  refine norm_id_of_nontrivial_seminorm V ?_
  /-
    V : Type u_5
    inst✝¹ : NormedAddCommGroup V
    inst✝ : Nontrivial V
    ⊢ Exists fun x => Ne (Norm.norm x) 0
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : V)
  /-
    case intro
    V : Type u_5
    inst✝¹ : NormedAddCommGroup V
    inst✝ : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Exists fun x => Ne (Norm.norm x) 0
  -/
  exact ⟨x, ne_of_gt (norm_pos_iff.2 hx)⟩
  /-
    🎉 no goals
  -/


theorem coe_id : (NormedAddGroupHom.id V : V → V) = _root_.id :=
  rfl


/-- Opposite of a normed group hom. -/
instance neg : Neg (NormedAddGroupHom V₁ V₂) :=
                                                                    /-
                                                                      V : Type u_1
                                                                      V₁ : Type u_2
                                                                      V₂ : Type u_3
                                                                      V₃ : Type u_4
                                                                      inst✝³ : SeminormedAddCommGroup V
                                                                      inst✝² : SeminormedAddCommGroup V₁
                                                                      inst✝¹ : SeminormedAddCommGroup V₂
                                                                      inst✝ : SeminormedAddCommGroup V₃
                                                                      f✝ g f : NormedAddGroupHom V₁ V₂
                                                                      v : V₁
                                                                      ⊢ LE.le (Norm.norm ((Neg.neg f.toAddMonoidHom) v)) (HMul.hMul (Norm.norm f) (N …
                                                                    -/
  ⟨fun f => (-f.toAddMonoidHom).mkNormedAddGroupHom ‖f‖ fun v => by simp [le_opNorm f v]⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem coe_neg (f : NormedAddGroupHom V₁ V₂) : ⇑(-f) = -f :=
  rfl


@[simp]
theorem neg_apply (f : NormedAddGroupHom V₁ V₂) (v : V₁) :
    (-f : NormedAddGroupHom V₁ V₂) v = -f v :=
  rfl


theorem opNorm_neg (f : NormedAddGroupHom V₁ V₂) : ‖-f‖ = ‖f‖ := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    ⊢ Eq (Norm.norm (Neg.neg f)) (Norm.norm f)
  -/
  simp only [norm_def, coe_neg, norm_neg, Pi.neg_apply]
  /-
    🎉 no goals
  -/


/-- Subtraction of normed group homs. -/
instance sub : Sub (NormedAddGroupHom V₁ V₂) :=
  ⟨fun f g =>
    { f.toAddMonoidHom - g.toAddMonoidHom with
      bound' := by
        /-
          V : Type u_1
          V₁ : Type u_2
          V₂ : Type u_3
          V₃ : Type u_4
          inst✝³ : SeminormedAddCommGroup V
          inst✝² : SeminormedAddCommGroup V₁
          inst✝¹ : SeminormedAddCommGroup V₂
          inst✝ : SeminormedAddCommGroup V₃
          f✝ g✝ f g : NormedAddGroupHom V₁ V₂
          ⊢ Exists fun C => ∀ (v : V₁), LE.le (Norm.norm ((↑__src✝).toFun v)) (HMul.hMul …
        -/
        simp only [AddMonoidHom.sub_apply, AddMonoidHom.toFun_eq_coe, sub_eq_add_neg]
        /-
          V : Type u_1
          V₁ : Type u_2
          V₂ : Type u_3
          V₃ : Type u_4
          inst✝³ : SeminormedAddCommGroup V
          inst✝² : SeminormedAddCommGroup V₁
          inst✝¹ : SeminormedAddCommGroup V₂
          inst✝ : SeminormedAddCommGroup V₃
          f✝ g✝ f g : NormedAddGroupHom V₁ V₂
          ⊢ Exists fun C => ∀ (v : V₁), LE.le (Norm.norm ((HAdd.hAdd f.toAddMonoidHom (N …
        -/
        exact (f + -g).bound' }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_sub (f g : NormedAddGroupHom V₁ V₂) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem sub_apply (f g : NormedAddGroupHom V₁ V₂) (v : V₁) :
    (f - g : NormedAddGroupHom V₁ V₂) v = f v - g v :=
  rfl


instance smul : SMul R (NormedAddGroupHom V₁ V₂) where
  smul r f :=
    { toFun := r • ⇑f
      map_add' := (r • f.toAddMonoidHom).map_add'
      bound' :=
        let ⟨b, hb⟩ := f.bound'
        ⟨dist r 0 * b, fun x => by
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝¹¹ : SeminormedAddCommGroup V
            inst✝¹⁰ : SeminormedAddCommGroup V₁
            inst✝⁹ : SeminormedAddCommGroup V₂
            inst✝⁸ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            R : Type u_5
            R' : Type u_6
            inst✝⁷ : MonoidWithZero R
            inst✝⁶ : DistribMulAction R V₂
            inst✝⁵ : PseudoMetricSpace R
            inst✝⁴ : BoundedSMul R V₂
            inst✝³ : MonoidWithZero R'
            inst✝² : DistribMulAction R' V₂
            inst✝¹ : PseudoMetricSpace R'
            inst✝ : BoundedSMul R' V₂
            r : R
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            x : V₁
            ⊢ LE.le (Norm.norm (HSMul.hSMul r (⇑f) x)) (HMul.hMul (HMul.hMul (Dist.dist r  …
          -/
          have := dist_smul_pair r (f x) (f 0)
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝¹¹ : SeminormedAddCommGroup V
            inst✝¹⁰ : SeminormedAddCommGroup V₁
            inst✝⁹ : SeminormedAddCommGroup V₂
            inst✝⁸ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            R : Type u_5
            R' : Type u_6
            inst✝⁷ : MonoidWithZero R
            inst✝⁶ : DistribMulAction R V₂
            inst✝⁵ : PseudoMetricSpace R
            inst✝⁴ : BoundedSMul R V₂
            inst✝³ : MonoidWithZero R'
            inst✝² : DistribMulAction R' V₂
            inst✝¹ : PseudoMetricSpace R'
            inst✝ : BoundedSMul R' V₂
            r : R
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            x : V₁
            this : LE.le (Dist.dist (HSMul.hSMul r (f x)) (HSMul.hSMul r (f 0))) (HMul.hMu …
            ⊢ LE.le (Norm.norm (HSMul.hSMul r (⇑f) x)) (HMul.hMul (HMul.hMul (Dist.dist r  …
          -/
          rw [map_zero, smul_zero, dist_zero_right, dist_zero_right] at this
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝¹¹ : SeminormedAddCommGroup V
            inst✝¹⁰ : SeminormedAddCommGroup V₁
            inst✝⁹ : SeminormedAddCommGroup V₂
            inst✝⁸ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            R : Type u_5
            R' : Type u_6
            inst✝⁷ : MonoidWithZero R
            inst✝⁶ : DistribMulAction R V₂
            inst✝⁵ : PseudoMetricSpace R
            inst✝⁴ : BoundedSMul R V₂
            inst✝³ : MonoidWithZero R'
            inst✝² : DistribMulAction R' V₂
            inst✝¹ : PseudoMetricSpace R'
            inst✝ : BoundedSMul R' V₂
            r : R
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            x : V₁
            this : LE.le (Norm.norm (HSMul.hSMul r (f x))) (HMul.hMul (Dist.dist r 0) (Nor …
            ⊢ LE.le (Norm.norm (HSMul.hSMul r (⇑f) x)) (HMul.hMul (HMul.hMul (Dist.dist r  …
          -/
          rw [mul_assoc]
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝¹¹ : SeminormedAddCommGroup V
            inst✝¹⁰ : SeminormedAddCommGroup V₁
            inst✝⁹ : SeminormedAddCommGroup V₂
            inst✝⁸ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            R : Type u_5
            R' : Type u_6
            inst✝⁷ : MonoidWithZero R
            inst✝⁶ : DistribMulAction R V₂
            inst✝⁵ : PseudoMetricSpace R
            inst✝⁴ : BoundedSMul R V₂
            inst✝³ : MonoidWithZero R'
            inst✝² : DistribMulAction R' V₂
            inst✝¹ : PseudoMetricSpace R'
            inst✝ : BoundedSMul R' V₂
            r : R
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            x : V₁
            this : LE.le (Norm.norm (HSMul.hSMul r (f x))) (HMul.hMul (Dist.dist r 0) (Nor …
            ⊢ LE.le (Norm.norm (HSMul.hSMul r (⇑f) x)) (HMul.hMul (Dist.dist r 0) (HMul.hM …
          -/
          refine this.trans ?_
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝¹¹ : SeminormedAddCommGroup V
            inst✝¹⁰ : SeminormedAddCommGroup V₁
            inst✝⁹ : SeminormedAddCommGroup V₂
            inst✝⁸ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            R : Type u_5
            R' : Type u_6
            inst✝⁷ : MonoidWithZero R
            inst✝⁶ : DistribMulAction R V₂
            inst✝⁵ : PseudoMetricSpace R
            inst✝⁴ : BoundedSMul R V₂
            inst✝³ : MonoidWithZero R'
            inst✝² : DistribMulAction R' V₂
            inst✝¹ : PseudoMetricSpace R'
            inst✝ : BoundedSMul R' V₂
            r : R
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            x : V₁
            this : LE.le (Norm.norm (HSMul.hSMul r (f x))) (HMul.hMul (Dist.dist r 0) (Nor …
            ⊢ LE.le (HMul.hMul (Dist.dist r 0) (Norm.norm (f x))) (HMul.hMul (Dist.dist r  …
          -/
          gcongr
          /-
            case h
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝¹¹ : SeminormedAddCommGroup V
            inst✝¹⁰ : SeminormedAddCommGroup V₁
            inst✝⁹ : SeminormedAddCommGroup V₂
            inst✝⁸ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            R : Type u_5
            R' : Type u_6
            inst✝⁷ : MonoidWithZero R
            inst✝⁶ : DistribMulAction R V₂
            inst✝⁵ : PseudoMetricSpace R
            inst✝⁴ : BoundedSMul R V₂
            inst✝³ : MonoidWithZero R'
            inst✝² : DistribMulAction R' V₂
            inst✝¹ : PseudoMetricSpace R'
            inst✝ : BoundedSMul R' V₂
            r : R
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            x : V₁
            this : LE.le (Norm.norm (HSMul.hSMul r (f x))) (HMul.hMul (Dist.dist r 0) (Nor …
            ⊢ LE.le (Norm.norm (f x)) (HMul.hMul b (Norm.norm x))
          -/
          exact hb x⟩ }
          /-
            🎉 no goals
          -/


@[simp]
theorem coe_smul (r : R) (f : NormedAddGroupHom V₁ V₂) : ⇑(r • f) = r • ⇑f :=
  rfl


@[simp]
theorem smul_apply (r : R) (f : NormedAddGroupHom V₁ V₂) (v : V₁) : (r • f) v = r • f v :=
  rfl


instance smulCommClass [SMulCommClass R R' V₂] :
    SMulCommClass R R' (NormedAddGroupHom V₁ V₂) where
  smul_comm _ _ _ := ext fun _ => smul_comm _ _ _


instance isScalarTower [SMul R R'] [IsScalarTower R R' V₂] :
    IsScalarTower R R' (NormedAddGroupHom V₁ V₂) where
  smul_assoc _ _ _ := ext fun _ => smul_assoc _ _ _


instance isCentralScalar [DistribMulAction Rᵐᵒᵖ V₂] [IsCentralScalar R V₂] :
    IsCentralScalar R (NormedAddGroupHom V₁ V₂) where
  op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


instance nsmul : SMul ℕ (NormedAddGroupHom V₁ V₂) where
  smul n f :=
    { toFun := n • ⇑f
      map_add' := (n • f.toAddMonoidHom).map_add'
      bound' :=
        let ⟨b, hb⟩ := f.bound'
        ⟨n • b, fun v => by
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            n : Nat
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            v : V₁
            ⊢ LE.le (Norm.norm (HSMul.hSMul n (⇑f) v)) (HMul.hMul (HSMul.hSMul n b) (Norm. …
          -/
          rw [Pi.smul_apply, nsmul_eq_mul, mul_assoc]
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            n : Nat
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            v : V₁
            ⊢ LE.le (Norm.norm (HSMul.hSMul n (f v))) (HMul.hMul (↑n) (HMul.hMul b (Norm.n …
          -/
          exact norm_nsmul_le.trans (by gcongr; apply hb)⟩ }
          /-
            🎉 no goals
          -/


@[simp]
theorem coe_nsmul (r : ℕ) (f : NormedAddGroupHom V₁ V₂) : ⇑(r • f) = r • ⇑f :=
  rfl


@[simp]
theorem nsmul_apply (r : ℕ) (f : NormedAddGroupHom V₁ V₂) (v : V₁) : (r • f) v = r • f v :=
  rfl


instance zsmul : SMul ℤ (NormedAddGroupHom V₁ V₂) where
  smul z f :=
    { toFun := z • ⇑f
      map_add' := (z • f.toAddMonoidHom).map_add'
      bound' :=
        let ⟨b, hb⟩ := f.bound'
        ⟨‖z‖ • b, fun v => by
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            z : Int
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            v : V₁
            ⊢ LE.le (Norm.norm (HSMul.hSMul z (⇑f) v)) (HMul.hMul (HSMul.hSMul (Norm.norm  …
          -/
          rw [Pi.smul_apply, smul_eq_mul, mul_assoc]
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f✝ g : NormedAddGroupHom V₁ V₂
            z : Int
            f : NormedAddGroupHom V₁ V₂
            b : Real
            hb : ∀ (v : V₁), LE.le (Norm.norm (f.toFun v)) (HMul.hMul b (Norm.norm v))
            v : V₁
            ⊢ LE.le (Norm.norm (HSMul.hSMul z (f v))) (HMul.hMul (Norm.norm z) (HMul.hMul  …
          -/
          exact (norm_zsmul_le _ _).trans (by gcongr; apply hb)⟩ }
          /-
            🎉 no goals
          -/


@[simp]
theorem coe_zsmul (r : ℤ) (f : NormedAddGroupHom V₁ V₂) : ⇑(r • f) = r • ⇑f :=
  rfl


@[simp]
theorem zsmul_apply (r : ℤ) (f : NormedAddGroupHom V₁ V₂) (v : V₁) : (r • f) v = r • f v :=
  rfl


/-- Homs between two given normed groups form a commutative additive group. -/
instance toAddCommGroup : AddCommGroup (NormedAddGroupHom V₁ V₂) :=
  coe_injective.addCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    fun _ _ => rfl


/-- Normed group homomorphisms themselves form a seminormed group with respect to
    the operator norm. -/
instance toSeminormedAddCommGroup : SeminormedAddCommGroup (NormedAddGroupHom V₁ V₂) :=
  AddGroupSeminorm.toSeminormedAddCommGroup
    { toFun := opNorm
      map_zero' := opNorm_zero
      neg' := opNorm_neg
      add_le' := opNorm_add_le }


/-- Normed group homomorphisms themselves form a normed group with respect to
    the operator norm. -/
instance toNormedAddCommGroup {V₁ V₂ : Type*} [NormedAddCommGroup V₁] [NormedAddCommGroup V₂] :
    NormedAddCommGroup (NormedAddGroupHom V₁ V₂) :=
  AddGroupNorm.toNormedAddCommGroup
    { toFun := opNorm
      map_zero' := opNorm_zero
      neg' := opNorm_neg
      add_le' := opNorm_add_le
      eq_zero_of_map_eq_zero' := fun _f => opNorm_zero_iff.1 }


/-- Coercion of a `NormedAddGroupHom` is an `AddMonoidHom`. Similar to `AddMonoidHom.coeFn`. -/
@[simps]
def coeAddHom : NormedAddGroupHom V₁ V₂ →+ V₁ → V₂ where
  toFun := DFunLike.coe
  map_zero' := coe_zero
  map_add' := coe_add


@[simp]
theorem coe_sum {ι : Type*} (s : Finset ι) (f : ι → NormedAddGroupHom V₁ V₂) :
    ⇑(∑ i ∈ s, f i) = ∑ i ∈ s, (f i : V₁ → V₂) :=
  map_sum coeAddHom f s


theorem sum_apply {ι : Type*} (s : Finset ι) (f : ι → NormedAddGroupHom V₁ V₂) (v : V₁) :
                                            /-
                                              V₁ : Type u_2
                                              V₂ : Type u_3
                                              inst✝¹ : SeminormedAddCommGroup V₁
                                              inst✝ : SeminormedAddCommGroup V₂
                                              ι : Type u_5
                                              s : Finset ι
                                              f : ι → NormedAddGroupHom V₁ V₂
                                              v : V₁
                                              ⊢ Eq ((s.sum fun i => f i) v) (s.sum fun i => (f i) v)
                                            -/
    (∑ i ∈ s, f i) v = ∑ i ∈ s, f i v := by simp only [coe_sum, Finset.sum_apply]
                                            /-
                                              🎉 no goals
                                            -/


instance distribMulAction {R : Type*} [MonoidWithZero R] [DistribMulAction R V₂]
    [PseudoMetricSpace R] [BoundedSMul R V₂] : DistribMulAction R (NormedAddGroupHom V₁ V₂) :=
  Function.Injective.distribMulAction coeAddHom coe_injective coe_smul


instance module {R : Type*} [Semiring R] [Module R V₂] [PseudoMetricSpace R] [BoundedSMul R V₂] :
    Module R (NormedAddGroupHom V₁ V₂) :=
  Function.Injective.module _ coeAddHom coe_injective coe_smul


/-- The composition of continuous normed group homs. -/
@[simps!]
protected def comp (g : NormedAddGroupHom V₂ V₃) (f : NormedAddGroupHom V₁ V₂) :
    NormedAddGroupHom V₁ V₃ :=
  (g.toAddMonoidHom.comp f.toAddMonoidHom).mkNormedAddGroupHom (‖g‖ * ‖f‖) fun v =>
    calc
      ‖g (f v)‖ ≤ ‖g‖ * ‖f v‖ := le_opNorm _ _
                                  /-
                                    V : Type u_1
                                    V₁ : Type u_2
                                    V₂ : Type u_3
                                    V₃ : Type u_4
                                    inst✝³ : SeminormedAddCommGroup V
                                    inst✝² : SeminormedAddCommGroup V₁
                                    inst✝¹ : SeminormedAddCommGroup V₂
                                    inst✝ : SeminormedAddCommGroup V₃
                                    f✝ g✝ : NormedAddGroupHom V₁ V₂
                                    g : NormedAddGroupHom V₂ V₃
                                    f : NormedAddGroupHom V₁ V₂
                                    v : V₁
                                    ⊢ LE.le (HMul.hMul (Norm.norm g) (Norm.norm (f v))) (HMul.hMul (Norm.norm g) ( …
                                  -/
      _ ≤ ‖g‖ * (‖f‖ * ‖v‖) := by gcongr; apply le_opNorm
                                          /-
                                            🎉 no goals
                                          -/
                                /-
                                  V : Type u_1
                                  V₁ : Type u_2
                                  V₂ : Type u_3
                                  V₃ : Type u_4
                                  inst✝³ : SeminormedAddCommGroup V
                                  inst✝² : SeminormedAddCommGroup V₁
                                  inst✝¹ : SeminormedAddCommGroup V₂
                                  inst✝ : SeminormedAddCommGroup V₃
                                  f✝ g✝ : NormedAddGroupHom V₁ V₂
                                  g : NormedAddGroupHom V₂ V₃
                                  f : NormedAddGroupHom V₁ V₂
                                  v : V₁
                                  ⊢ Eq (HMul.hMul (Norm.norm g) (HMul.hMul (Norm.norm f) (Norm.norm v))) (HMul.h …
                                -/
      _ = ‖g‖ * ‖f‖ * ‖v‖ := by rw [mul_assoc]
                                /-
                                  🎉 no goals
                                -/


theorem norm_comp_le (g : NormedAddGroupHom V₂ V₃) (f : NormedAddGroupHom V₁ V₂) :
    ‖g.comp f‖ ≤ ‖g‖ * ‖f‖ :=
  mkNormedAddGroupHom_norm_le _ (mul_nonneg (opNorm_nonneg _) (opNorm_nonneg _)) _


theorem norm_comp_le_of_le {g : NormedAddGroupHom V₂ V₃} {C₁ C₂ : ℝ} (hg : ‖g‖ ≤ C₂)
    (hf : ‖f‖ ≤ C₁) : ‖g.comp f‖ ≤ C₂ * C₁ :=
                                    /-
                                      V₁ : Type u_2
                                      V₂ : Type u_3
                                      V₃ : Type u_4
                                      inst✝² : SeminormedAddCommGroup V₁
                                      inst✝¹ : SeminormedAddCommGroup V₂
                                      inst✝ : SeminormedAddCommGroup V₃
                                      f : NormedAddGroupHom V₁ V₂
                                      g : NormedAddGroupHom V₂ V₃
                                      C₁ C₂ : Real
                                      hg : LE.le (Norm.norm g) C₂
                                      hf : LE.le (Norm.norm f) C₁
                                      ⊢ LE.le (HMul.hMul (Norm.norm g) (Norm.norm f)) (HMul.hMul C₂ C₁)
                                    -/
  le_trans (norm_comp_le g f) <| by gcongr; exact le_trans (norm_nonneg _) hg
                                            /-
                                              🎉 no goals
                                            -/


theorem norm_comp_le_of_le' {g : NormedAddGroupHom V₂ V₃} (C₁ C₂ C₃ : ℝ) (h : C₃ = C₂ * C₁)
    (hg : ‖g‖ ≤ C₂) (hf : ‖f‖ ≤ C₁) : ‖g.comp f‖ ≤ C₃ := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    g : NormedAddGroupHom V₂ V₃
    C₁ C₂ C₃ : Real
    h : Eq C₃ (HMul.hMul C₂ C₁)
    hg : LE.le (Norm.norm g) C₂
    hf : LE.le (Norm.norm f) C₁
    ⊢ LE.le (Norm.norm (g.comp f)) C₃
  -/
  rw [h]
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    g : NormedAddGroupHom V₂ V₃
    C₁ C₂ C₃ : Real
    h : Eq C₃ (HMul.hMul C₂ C₁)
    hg : LE.le (Norm.norm g) C₂
    hf : LE.le (Norm.norm f) C₁
    ⊢ LE.le (Norm.norm (g.comp f)) (HMul.hMul C₂ C₁)
  -/
  exact norm_comp_le_of_le hg hf
  /-
    🎉 no goals
  -/


/-- Composition of normed groups hom as an additive group morphism. -/
def compHom : NormedAddGroupHom V₂ V₃ →+ NormedAddGroupHom V₁ V₂ →+ NormedAddGroupHom V₁ V₃ :=
  AddMonoidHom.mk'
    (fun g =>
      AddMonoidHom.mk' (fun f => g.comp f)
        (by
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f g✝ : NormedAddGroupHom V₁ V₂
            g : NormedAddGroupHom V₂ V₃
            ⊢ ∀ (a b : NormedAddGroupHom V₁ V₂), Eq ((fun f => g.comp f) (HAdd.hAdd a b))  …
          -/
          intros
          /-
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f g✝ : NormedAddGroupHom V₁ V₂
            g : NormedAddGroupHom V₂ V₃
            a✝ b✝ : NormedAddGroupHom V₁ V₂
            ⊢ Eq ((fun f => g.comp f) (HAdd.hAdd a✝ b✝)) (HAdd.hAdd ((fun f => g.comp f) a …
          -/
          ext
          /-
            case H
            V : Type u_1
            V₁ : Type u_2
            V₂ : Type u_3
            V₃ : Type u_4
            inst✝³ : SeminormedAddCommGroup V
            inst✝² : SeminormedAddCommGroup V₁
            inst✝¹ : SeminormedAddCommGroup V₂
            inst✝ : SeminormedAddCommGroup V₃
            f g✝ : NormedAddGroupHom V₁ V₂
            g : NormedAddGroupHom V₂ V₃
            a✝ b✝ : NormedAddGroupHom V₁ V₂
            x✝ : V₁
            ⊢ Eq (((fun f => g.comp f) (HAdd.hAdd a✝ b✝)) x✝) ((HAdd.hAdd ((fun f => g.com …
          -/
          exact map_add g _ _))
          /-
            🎉 no goals
          -/
    (by
      /-
        V : Type u_1
        V₁ : Type u_2
        V₂ : Type u_3
        V₃ : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : SeminormedAddCommGroup V₁
        inst✝¹ : SeminormedAddCommGroup V₂
        inst✝ : SeminormedAddCommGroup V₃
        f g : NormedAddGroupHom V₁ V₂
        ⊢ ∀ (a b : NormedAddGroupHom V₂ V₃), Eq ((fun g => AddMonoidHom.mk' (fun f =>  …
      -/
      intros
      /-
        V : Type u_1
        V₁ : Type u_2
        V₂ : Type u_3
        V₃ : Type u_4
        inst✝³ : SeminormedAddCommGroup V
        inst✝² : SeminormedAddCommGroup V₁
        inst✝¹ : SeminormedAddCommGroup V₂
        inst✝ : SeminormedAddCommGroup V₃
        f g : NormedAddGroupHom V₁ V₂
        a✝ b✝ : NormedAddGroupHom V₂ V₃
        ⊢ Eq ((fun g => AddMonoidHom.mk' (fun f => g.comp f) ⋯) (HAdd.hAdd a✝ b✝)) (HA …
      -/
      ext
      simp only [comp_apply, Pi.add_apply, Function.comp_apply, AddMonoidHom.add_apply,
        AddMonoidHom.mk'_apply, coe_add])


@[simp]
theorem comp_zero (f : NormedAddGroupHom V₂ V₃) : f.comp (0 : NormedAddGroupHom V₁ V₂) = 0 := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₂ V₃
    ⊢ Eq (f.comp 0) 0
  -/
  ext
  /-
    case H
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₂ V₃
    x✝ : V₁
    ⊢ Eq ((f.comp 0) x✝) (0 x✝)
  -/
  exact map_zero f
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_comp (f : NormedAddGroupHom V₁ V₂) : (0 : NormedAddGroupHom V₂ V₃).comp f = 0 := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    ⊢ Eq (NormedAddGroupHom.comp 0 f) 0
  -/
  ext
  /-
    case H
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    x✝ : V₁
    ⊢ Eq ((NormedAddGroupHom.comp 0 f) x✝) (0 x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comp_assoc {V₄ : Type*} [SeminormedAddCommGroup V₄] (h : NormedAddGroupHom V₃ V₄)
    (g : NormedAddGroupHom V₂ V₃) (f : NormedAddGroupHom V₁ V₂) :
    (h.comp g).comp f = h.comp (g.comp f) := by
  /-
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝³ : SeminormedAddCommGroup V₁
    inst✝² : SeminormedAddCommGroup V₂
    inst✝¹ : SeminormedAddCommGroup V₃
    V₄ : Type u_5
    inst✝ : SeminormedAddCommGroup V₄
    h : NormedAddGroupHom V₃ V₄
    g : NormedAddGroupHom V₂ V₃
    f : NormedAddGroupHom V₁ V₂
    ⊢ Eq ((h.comp g).comp f) (h.comp (g.comp f))
  -/
  ext
  /-
    case H
    V₁ : Type u_2
    V₂ : Type u_3
    V₃ : Type u_4
    inst✝³ : SeminormedAddCommGroup V₁
    inst✝² : SeminormedAddCommGroup V₂
    inst✝¹ : SeminormedAddCommGroup V₃
    V₄ : Type u_5
    inst✝ : SeminormedAddCommGroup V₄
    h : NormedAddGroupHom V₃ V₄
    g : NormedAddGroupHom V₂ V₃
    f : NormedAddGroupHom V₁ V₂
    x✝ : V₁
    ⊢ Eq (((h.comp g).comp f) x✝) ((h.comp (g.comp f)) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coe_comp (f : NormedAddGroupHom V₁ V₂) (g : NormedAddGroupHom V₂ V₃) :
    (g.comp f : V₁ → V₃) = (g : V₂ → V₃) ∘ (f : V₁ → V₂) :=
  rfl


/-- The inclusion of an `AddSubgroup`, as bounded group homomorphism. -/
@[simps!]
def incl (s : AddSubgroup V) : NormedAddGroupHom s V where
  toFun := (Subtype.val : s → V)
  map_add' _ _ := AddSubgroup.coe_add _ _ _
                            /-
                              V : Type u_1
                              W : Type u_2
                              V₁ : Type u_3
                              V₂ : Type u_4
                              V₃ : Type u_5
                              inst✝⁴ : SeminormedAddCommGroup V
                              inst✝³ : SeminormedAddCommGroup W
                              inst✝² : SeminormedAddCommGroup V₁
                              inst✝¹ : SeminormedAddCommGroup V₂
                              inst✝ : SeminormedAddCommGroup V₃
                              s : AddSubgroup V
                              v : Subtype fun x => Membership.mem s x
                              ⊢ LE.le (Norm.norm ↑v) (HMul.hMul 1 (Norm.norm v))
                            -/
  bound' := ⟨1, fun v => by rw [one_mul, AddSubgroup.coe_norm]⟩
                            /-
                              🎉 no goals
                            -/


theorem norm_incl {V' : AddSubgroup V} (x : V') : ‖incl _ x‖ = ‖x‖ :=
  rfl


/-- The kernel of a bounded group homomorphism. Naturally endowed with a
`SeminormedAddCommGroup` instance. -/
def ker : AddSubgroup V₁ :=
  f.toAddMonoidHom.ker


theorem mem_ker (v : V₁) : v ∈ f.ker ↔ f v = 0 := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    v : V₁
    ⊢ Iff (Membership.mem f.ker v) (Eq (f v) 0)
  -/
  erw [f.toAddMonoidHom.mem_ker, coe_toAddMonoidHom]
  /-
    🎉 no goals
  -/


/-- Given a normed group hom `f : V₁ → V₂` satisfying `g.comp f = 0` for some `g : V₂ → V₃`,
    the corestriction of `f` to the kernel of `g`. -/
@[simps]
def ker.lift (h : g.comp f = 0) : NormedAddGroupHom V₁ g.ker where
                      /-
                        V : Type u_1
                        W : Type u_2
                        V₁ : Type u_3
                        V₂ : Type u_4
                        V₃ : Type u_5
                        inst✝⁴ : SeminormedAddCommGroup V
                        inst✝³ : SeminormedAddCommGroup W
                        inst✝² : SeminormedAddCommGroup V₁
                        inst✝¹ : SeminormedAddCommGroup V₂
                        inst✝ : SeminormedAddCommGroup V₃
                        f : NormedAddGroupHom V₁ V₂
                        g : NormedAddGroupHom V₂ V₃
                        h : Eq (g.comp f) 0
                        v : V₁
                        ⊢ Membership.mem g.ker (f v)
                      -/
  toFun v := ⟨f v, by rw [g.mem_ker, ← comp_apply g f, h, zero_apply]⟩
                      /-
                        🎉 no goals
                      -/
                     /-
                       V : Type u_1
                       W : Type u_2
                       V₁ : Type u_3
                       V₂ : Type u_4
                       V₃ : Type u_5
                       inst✝⁴ : SeminormedAddCommGroup V
                       inst✝³ : SeminormedAddCommGroup W
                       inst✝² : SeminormedAddCommGroup V₁
                       inst✝¹ : SeminormedAddCommGroup V₂
                       inst✝ : SeminormedAddCommGroup V₃
                       f : NormedAddGroupHom V₁ V₂
                       g : NormedAddGroupHom V₂ V₃
                       h : Eq (g.comp f) 0
                       v w : V₁
                       ⊢ Eq ((fun v => ⟨f v, ⋯⟩) (HAdd.hAdd v w)) (HAdd.hAdd ((fun v => ⟨f v, ⋯⟩) v)  …
                     -/
  map_add' v w := by simp only [map_add, AddMemClass.mk_add_mk]
                     /-
                       🎉 no goals
                     -/
  bound' := f.bound'


@[simp]
theorem ker.incl_comp_lift (h : g.comp f = 0) : (incl g.ker).comp (ker.lift f g h) = f := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    g : NormedAddGroupHom V₂ V₃
    h : Eq (g.comp f) 0
    ⊢ Eq ((NormedAddGroupHom.incl g.ker).comp (NormedAddGroupHom.ker.lift f g h)) f
  -/
  ext
  /-
    case H
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    g : NormedAddGroupHom V₂ V₃
    h : Eq (g.comp f) 0
    x✝ : V₁
    ⊢ Eq (((NormedAddGroupHom.incl g.ker).comp (NormedAddGroupHom.ker.lift f g h)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ker_zero : (0 : NormedAddGroupHom V₁ V₂).ker = ⊤ := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    ⊢ Eq (NormedAddGroupHom.ker 0) Top.top
  -/
  ext
  /-
    case h
    V₁ : Type u_3
    V₂ : Type u_4
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    x✝ : V₁
    ⊢ Iff (Membership.mem (NormedAddGroupHom.ker 0) x✝) (Membership.mem Top.top x✝)
  -/
  simp [mem_ker]
  /-
    🎉 no goals
  -/


theorem coe_ker : (f.ker : Set V₁) = (f : V₁ → V₂) ⁻¹' {0} :=
  rfl


theorem isClosed_ker {V₂ : Type*} [NormedAddCommGroup V₂] (f : NormedAddGroupHom V₁ V₂) :
    IsClosed (f.ker : Set V₁) :=
  f.coe_ker ▸ IsClosed.preimage f.continuous (T1Space.t1 0)


/-- The image of a bounded group homomorphism. Naturally endowed with a
`SeminormedAddCommGroup` instance. -/
def range : AddSubgroup V₂ :=
  f.toAddMonoidHom.range


theorem mem_range (v : V₂) : v ∈ f.range ↔ ∃ w, f w = v := Iff.rfl


@[simp]
theorem mem_range_self (v : V₁) : f v ∈ f.range :=
  ⟨v, rfl⟩


theorem comp_range : (g.comp f).range = AddSubgroup.map g.toAddMonoidHom f.range := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    g : NormedAddGroupHom V₂ V₃
    ⊢ Eq (g.comp f).range (AddSubgroup.map g.toAddMonoidHom f.range)
  -/
  erw [AddMonoidHom.map_range]
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝² : SeminormedAddCommGroup V₁
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : SeminormedAddCommGroup V₃
    f : NormedAddGroupHom V₁ V₂
    g : NormedAddGroupHom V₂ V₃
    ⊢ Eq (g.comp f).range (g.toAddMonoidHom.comp f.toAddMonoidHom).range
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem incl_range (s : AddSubgroup V₁) : (incl s).range = s := by
  /-
    V₁ : Type u_3
    inst✝ : SeminormedAddCommGroup V₁
    s : AddSubgroup V₁
    ⊢ Eq (NormedAddGroupHom.incl s).range s
  -/
  ext x
  /-
    case h
    V₁ : Type u_3
    inst✝ : SeminormedAddCommGroup V₁
    s : AddSubgroup V₁
    x : V₁
    ⊢ Iff (Membership.mem (NormedAddGroupHom.incl s).range x) (Membership.mem s x)
  -/
  exact ⟨fun ⟨y, hy⟩ => by rw [← hy]; simp, fun hx => ⟨⟨x, hx⟩, by simp⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem range_comp_incl_top : (f.comp (incl (⊤ : AddSubgroup V₁))).range = f.range := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    inst✝¹ : SeminormedAddCommGroup V₁
    inst✝ : SeminormedAddCommGroup V₂
    f : NormedAddGroupHom V₁ V₂
    ⊢ Eq (f.comp (NormedAddGroupHom.incl Top.top)).range f.range
  -/
  simp [comp_range, incl_range, ← AddMonoidHom.range_eq_map]; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- A `NormedAddGroupHom` is *norm-nonincreasing* if `‖f v‖ ≤ ‖v‖` for all `v`. -/
def NormNoninc (f : NormedAddGroupHom V W) : Prop :=
  ∀ v, ‖f v‖ ≤ ‖v‖


theorem normNoninc_iff_norm_le_one : f.NormNoninc ↔ ‖f‖ ≤ 1 := by
  /-
    V : Type u_1
    W : Type u_2
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : SeminormedAddCommGroup W
    f : NormedAddGroupHom V W
    ⊢ Iff f.NormNoninc (LE.le (Norm.norm f) 1)
  -/
  refine ⟨fun h => ?_, fun h => fun v => ?_⟩
    /-
      case refine_1
      V : Type u_1
      W : Type u_2
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : SeminormedAddCommGroup W
      f : NormedAddGroupHom V W
      h : f.NormNoninc
      ⊢ LE.le (Norm.norm f) 1
    -/
  · refine opNorm_le_bound _ zero_le_one fun v => ?_
    /-
      case refine_1
      V : Type u_1
      W : Type u_2
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : SeminormedAddCommGroup W
      f : NormedAddGroupHom V W
      h : f.NormNoninc
      v : V
      ⊢ LE.le (Norm.norm (f v)) (HMul.hMul 1 (Norm.norm v))
    -/
    simpa [one_mul] using h v
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u_1
      W : Type u_2
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : SeminormedAddCommGroup W
      f : NormedAddGroupHom V W
      h : LE.le (Norm.norm f) 1
      v : V
      ⊢ LE.le (Norm.norm (f v)) (Norm.norm v)
    -/
  · simpa using le_of_opNorm_le f h v
    /-
      🎉 no goals
    -/


                                                                       /-
                                                                         V₁ : Type u_3
                                                                         V₂ : Type u_4
                                                                         inst✝¹ : SeminormedAddCommGroup V₁
                                                                         inst✝ : SeminormedAddCommGroup V₂
                                                                         v : V₁
                                                                         ⊢ LE.le (Norm.norm (0 v)) (Norm.norm v)
                                                                       -/
theorem zero : (0 : NormedAddGroupHom V₁ V₂).NormNoninc := fun v => by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem id : (id V).NormNoninc := fun _v => le_rfl


theorem comp {g : NormedAddGroupHom V₂ V₃} {f : NormedAddGroupHom V₁ V₂} (hg : g.NormNoninc)
    (hf : f.NormNoninc) : (g.comp f).NormNoninc := fun v => (hg (f v)).trans (hf v)


@[simp]
theorem neg_iff {f : NormedAddGroupHom V₁ V₂} : (-f).NormNoninc ↔ f.NormNoninc :=
                 /-
                   V₁ : Type u_3
                   V₂ : Type u_4
                   inst✝¹ : SeminormedAddCommGroup V₁
                   inst✝ : SeminormedAddCommGroup V₂
                   f : NormedAddGroupHom V₁ V₂
                   h : (Neg.neg f).NormNoninc
                   x : V₁
                   ⊢ LE.le (Norm.norm (f x)) (Norm.norm x)
                 -/
  ⟨fun h x => by simpa using h x, fun h x => (norm_neg (f x)).le.trans (h x)⟩
                 /-
                   🎉 no goals
                 -/


theorem norm_eq_of_isometry {f : NormedAddGroupHom V W} (hf : Isometry f) (v : V) : ‖f v‖ = ‖v‖ :=
  (AddMonoidHomClass.isometry_iff_norm f).mp hf v


theorem isometry_id : @Isometry V V _ _ (id V) :=
  _root_.isometry_id


theorem isometry_comp {g : NormedAddGroupHom V₂ V₃} {f : NormedAddGroupHom V₁ V₂} (hg : Isometry g)
    (hf : Isometry f) : Isometry (g.comp f) :=
  hg.comp hf


theorem normNoninc_of_isometry (hf : Isometry f) : f.NormNoninc := fun v =>
  le_of_eq <| norm_eq_of_isometry hf v


/-- The equalizer of two morphisms `f g : NormedAddGroupHom V W`. -/
def equalizer :=
  (f - g).ker


/-- The inclusion of `f.equalizer g` as a `NormedAddGroupHom`. -/
def ι : NormedAddGroupHom (f.equalizer g) V :=
  incl _


theorem comp_ι_eq : f.comp (ι f g) = g.comp (ι f g) := by
  /-
    V : Type u_1
    W : Type u_2
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : SeminormedAddCommGroup W
    f g : NormedAddGroupHom V W
    ⊢ Eq (f.comp (NormedAddGroupHom.Equalizer.ι f g)) (g.comp (NormedAddGroupHom.E …
  -/
  ext x
  /-
    case H
    V : Type u_1
    W : Type u_2
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : SeminormedAddCommGroup W
    f g : NormedAddGroupHom V W
    x : Subtype fun x => Membership.mem (f.equalizer g) x
    ⊢ Eq ((f.comp (NormedAddGroupHom.Equalizer.ι f g)) x) ((g.comp (NormedAddGroup …
  -/
  rw [comp_apply, comp_apply, ← sub_eq_zero, ← NormedAddGroupHom.sub_apply]
  /-
    case H
    V : Type u_1
    W : Type u_2
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : SeminormedAddCommGroup W
    f g : NormedAddGroupHom V W
    x : Subtype fun x => Membership.mem (f.equalizer g) x
    ⊢ Eq ((HSub.hSub f g) ((NormedAddGroupHom.Equalizer.ι f g) x)) 0
  -/
  exact x.2
  /-
    🎉 no goals
  -/


/-- If `φ : NormedAddGroupHom V₁ V` is such that `f.comp φ = g.comp φ`, the induced morphism
`NormedAddGroupHom V₁ (f.equalizer g)`. -/
@[simps]
def lift (φ : NormedAddGroupHom V₁ V) (h : f.comp φ = g.comp φ) :
    NormedAddGroupHom V₁ (f.equalizer g) where
  toFun v :=
    ⟨φ v,
      show (f - g) (φ v) = 0 by
        /-
          V : Type u_1
          W : Type u_2
          V₁ : Type u_3
          V₂ : Type u_4
          V₃ : Type u_5
          inst✝⁷ : SeminormedAddCommGroup V
          inst✝⁶ : SeminormedAddCommGroup W
          inst✝⁵ : SeminormedAddCommGroup V₁
          inst✝⁴ : SeminormedAddCommGroup V₂
          inst✝³ : SeminormedAddCommGroup V₃
          f : NormedAddGroupHom V W
          W₁ : Type u_6
          W₂ : Type u_7
          W₃ : Type u_8
          inst✝² : SeminormedAddCommGroup W₁
          inst✝¹ : SeminormedAddCommGroup W₂
          inst✝ : SeminormedAddCommGroup W₃
          g : NormedAddGroupHom V W
          f₁ g₁ : NormedAddGroupHom V₁ W₁
          f₂ g₂ : NormedAddGroupHom V₂ W₂
          f₃ g₃ : NormedAddGroupHom V₃ W₃
          φ : NormedAddGroupHom V₁ V
          h : Eq (f.comp φ) (g.comp φ)
          v : V₁
          ⊢ Eq ((HSub.hSub f g) (φ v)) 0
        -/
        rw [NormedAddGroupHom.sub_apply, sub_eq_zero, ← comp_apply, h, comp_apply]⟩
        /-
          🎉 no goals
        -/
  map_add' v₁ v₂ := by
    /-
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      φ : NormedAddGroupHom V₁ V
      h : Eq (f.comp φ) (g.comp φ)
      v₁ v₂ : V₁
      ⊢ Eq ((fun v => ⟨φ v, ⋯⟩) (HAdd.hAdd v₁ v₂)) (HAdd.hAdd ((fun v => ⟨φ v, ⋯⟩) v …
    -/
    ext
    /-
      case a
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      φ : NormedAddGroupHom V₁ V
      h : Eq (f.comp φ) (g.comp φ)
      v₁ v₂ : V₁
      ⊢ Eq ↑((fun v => ⟨φ v, ⋯⟩) (HAdd.hAdd v₁ v₂)) ↑(HAdd.hAdd ((fun v => ⟨φ v, ⋯⟩) …
    -/
    simp only [map_add, AddSubgroup.coe_add, Subtype.coe_mk]
    /-
      🎉 no goals
    -/
  bound' := by
    /-
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      φ : NormedAddGroupHom V₁ V
      h : Eq (f.comp φ) (g.comp φ)
      ⊢ Exists fun C => ∀ (v : V₁), LE.le (Norm.norm ((fun v => ⟨φ v, ⋯⟩) v)) (HMul. …
    -/
    obtain ⟨C, _C_pos, hC⟩ := φ.bound
    /-
      case intro.intro
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      φ : NormedAddGroupHom V₁ V
      h : Eq (f.comp φ) (g.comp φ)
      C : Real
      _C_pos : LT.lt 0 C
      hC : ∀ (x : V₁), LE.le (Norm.norm (φ x)) (HMul.hMul C (Norm.norm x))
      ⊢ Exists fun C => ∀ (v : V₁), LE.le (Norm.norm ((fun v => ⟨φ v, ⋯⟩) v)) (HMul. …
    -/
    exact ⟨C, hC⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem ι_comp_lift (φ : NormedAddGroupHom V₁ V) (h : f.comp φ = g.comp φ) :
    (ι _ _).comp (lift φ h) = φ := by
  /-
    V : Type u_1
    W : Type u_2
    V₁ : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : SeminormedAddCommGroup W
    inst✝ : SeminormedAddCommGroup V₁
    f g : NormedAddGroupHom V W
    φ : NormedAddGroupHom V₁ V
    h : Eq (f.comp φ) (g.comp φ)
    ⊢ Eq ((NormedAddGroupHom.Equalizer.ι f g).comp (NormedAddGroupHom.Equalizer.li …
  -/
  ext
  /-
    case H
    V : Type u_1
    W : Type u_2
    V₁ : Type u_3
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : SeminormedAddCommGroup W
    inst✝ : SeminormedAddCommGroup V₁
    f g : NormedAddGroupHom V W
    φ : NormedAddGroupHom V₁ V
    h : Eq (f.comp φ) (g.comp φ)
    x✝ : V₁
    ⊢ Eq (((NormedAddGroupHom.Equalizer.ι f g).comp (NormedAddGroupHom.Equalizer.l …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The lifting property of the equalizer as an equivalence. -/
@[simps]
def liftEquiv :
    { φ : NormedAddGroupHom V₁ V // f.comp φ = g.comp φ } ≃
      NormedAddGroupHom V₁ (f.equalizer g) where
  toFun φ := lift φ φ.prop
                                  /-
                                    V : Type u_1
                                    W : Type u_2
                                    V₁ : Type u_3
                                    V₂ : Type u_4
                                    V₃ : Type u_5
                                    inst✝⁷ : SeminormedAddCommGroup V
                                    inst✝⁶ : SeminormedAddCommGroup W
                                    inst✝⁵ : SeminormedAddCommGroup V₁
                                    inst✝⁴ : SeminormedAddCommGroup V₂
                                    inst✝³ : SeminormedAddCommGroup V₃
                                    f : NormedAddGroupHom V W
                                    W₁ : Type u_6
                                    W₂ : Type u_7
                                    W₃ : Type u_8
                                    inst✝² : SeminormedAddCommGroup W₁
                                    inst✝¹ : SeminormedAddCommGroup W₂
                                    inst✝ : SeminormedAddCommGroup W₃
                                    g : NormedAddGroupHom V W
                                    f₁ g₁ : NormedAddGroupHom V₁ W₁
                                    f₂ g₂ : NormedAddGroupHom V₂ W₂
                                    f₃ g₃ : NormedAddGroupHom V₃ W₃
                                    ψ : NormedAddGroupHom V₁ (Subtype fun x => Membership.mem (f.equalizer g) x)
                                    ⊢ Eq (f.comp ((NormedAddGroupHom.Equalizer.ι f g).comp ψ)) (g.comp ((NormedAdd …
                                  -/
  invFun ψ := ⟨(ι f g).comp ψ, by rw [← comp_assoc, ← comp_assoc, comp_ι_eq]⟩
                                  /-
                                    🎉 no goals
                                  -/
                   /-
                     V : Type u_1
                     W : Type u_2
                     V₁ : Type u_3
                     V₂ : Type u_4
                     V₃ : Type u_5
                     inst✝⁷ : SeminormedAddCommGroup V
                     inst✝⁶ : SeminormedAddCommGroup W
                     inst✝⁵ : SeminormedAddCommGroup V₁
                     inst✝⁴ : SeminormedAddCommGroup V₂
                     inst✝³ : SeminormedAddCommGroup V₃
                     f : NormedAddGroupHom V W
                     W₁ : Type u_6
                     W₂ : Type u_7
                     W₃ : Type u_8
                     inst✝² : SeminormedAddCommGroup W₁
                     inst✝¹ : SeminormedAddCommGroup W₂
                     inst✝ : SeminormedAddCommGroup W₃
                     g : NormedAddGroupHom V W
                     f₁ g₁ : NormedAddGroupHom V₁ W₁
                     f₂ g₂ : NormedAddGroupHom V₂ W₂
                     f₃ g₃ : NormedAddGroupHom V₃ W₃
                     φ : Subtype fun φ => Eq (f.comp φ) (g.comp φ)
                     ⊢ Eq ((fun ψ => ⟨(NormedAddGroupHom.Equalizer.ι f g).comp ψ, ⋯⟩) ((fun φ => No …
                   -/
  left_inv φ := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv ψ := by
    /-
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      ψ : NormedAddGroupHom V₁ (Subtype fun x => Membership.mem (f.equalizer g) x)
      ⊢ Eq ((fun φ => NormedAddGroupHom.Equalizer.lift ↑φ ⋯) ((fun ψ => ⟨(NormedAddG …
    -/
    ext
    /-
      case H.a
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      ψ : NormedAddGroupHom V₁ (Subtype fun x => Membership.mem (f.equalizer g) x)
      x✝ : V₁
      ⊢ Eq ↑(((fun φ => NormedAddGroupHom.Equalizer.lift ↑φ ⋯) ((fun ψ => ⟨(NormedAd …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Given `φ : NormedAddGroupHom V₁ V₂` and `ψ : NormedAddGroupHom W₁ W₂` such that
`ψ.comp f₁ = f₂.comp φ` and `ψ.comp g₁ = g₂.comp φ`, the induced morphism
`NormedAddGroupHom (f₁.equalizer g₁) (f₂.equalizer g₂)`. -/
def map (φ : NormedAddGroupHom V₁ V₂) (ψ : NormedAddGroupHom W₁ W₂) (hf : ψ.comp f₁ = f₂.comp φ)
    (hg : ψ.comp g₁ = g₂.comp φ) : NormedAddGroupHom (f₁.equalizer g₁) (f₂.equalizer g₂) :=
  lift (φ.comp <| ι _ _) <| by
    /-
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      φ : NormedAddGroupHom V₁ V₂
      ψ : NormedAddGroupHom W₁ W₂
      hf : Eq (ψ.comp f₁) (f₂.comp φ)
      hg : Eq (ψ.comp g₁) (g₂.comp φ)
      ⊢ Eq (f₂.comp (φ.comp (NormedAddGroupHom.Equalizer.ι f₁ g₁))) (g₂.comp (φ.comp …
    -/
    simp only [← comp_assoc, ← hf, ← hg]
    /-
      V : Type u_1
      W : Type u_2
      V₁ : Type u_3
      V₂ : Type u_4
      V₃ : Type u_5
      inst✝⁷ : SeminormedAddCommGroup V
      inst✝⁶ : SeminormedAddCommGroup W
      inst✝⁵ : SeminormedAddCommGroup V₁
      inst✝⁴ : SeminormedAddCommGroup V₂
      inst✝³ : SeminormedAddCommGroup V₃
      f : NormedAddGroupHom V W
      W₁ : Type u_6
      W₂ : Type u_7
      W₃ : Type u_8
      inst✝² : SeminormedAddCommGroup W₁
      inst✝¹ : SeminormedAddCommGroup W₂
      inst✝ : SeminormedAddCommGroup W₃
      g : NormedAddGroupHom V W
      f₁ g₁ : NormedAddGroupHom V₁ W₁
      f₂ g₂ : NormedAddGroupHom V₂ W₂
      f₃ g₃ : NormedAddGroupHom V₃ W₃
      φ : NormedAddGroupHom V₁ V₂
      ψ : NormedAddGroupHom W₁ W₂
      hf : Eq (ψ.comp f₁) (f₂.comp φ)
      hg : Eq (ψ.comp g₁) (g₂.comp φ)
      ⊢ Eq ((ψ.comp f₁).comp (NormedAddGroupHom.Equalizer.ι f₁ g₁)) ((ψ.comp g₁).com …
    -/
    simp only [comp_assoc, comp_ι_eq f₁ g₁]
    /-
      🎉 no goals
    -/


@[simp]
theorem ι_comp_map (hf : ψ.comp f₁ = f₂.comp φ) (hg : ψ.comp g₁ = g₂.comp φ) :
    (ι f₂ g₂).comp (map φ ψ hf hg) = φ.comp (ι f₁ g₁) :=
  ι_comp_lift _ _


@[simp]
theorem map_id : map (f₂ := f₁) (g₂ := g₁) (id V₁) (id W₁) rfl rfl = id (f₁.equalizer g₁) := by
  /-
    V₁ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    W₁ : Type u_6
    inst✝ : SeminormedAddCommGroup W₁
    f₁ g₁ : NormedAddGroupHom V₁ W₁
    ⊢ Eq (NormedAddGroupHom.Equalizer.map (NormedAddGroupHom.id V₁) (NormedAddGrou …
  -/
  ext
  /-
    case H.a
    V₁ : Type u_3
    inst✝¹ : SeminormedAddCommGroup V₁
    W₁ : Type u_6
    inst✝ : SeminormedAddCommGroup W₁
    f₁ g₁ : NormedAddGroupHom V₁ W₁
    x✝ : Subtype fun x => Membership.mem (f₁.equalizer g₁) x
    ⊢ Eq ↑((NormedAddGroupHom.Equalizer.map (NormedAddGroupHom.id V₁) (NormedAddGr …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comm_sq₂ (hf : ψ.comp f₁ = f₂.comp φ) (hf' : ψ'.comp f₂ = f₃.comp φ') :
    (ψ'.comp ψ).comp f₁ = f₃.comp (φ'.comp φ) := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝⁵ : SeminormedAddCommGroup V₁
    inst✝⁴ : SeminormedAddCommGroup V₂
    inst✝³ : SeminormedAddCommGroup V₃
    W₁ : Type u_6
    W₂ : Type u_7
    W₃ : Type u_8
    inst✝² : SeminormedAddCommGroup W₁
    inst✝¹ : SeminormedAddCommGroup W₂
    inst✝ : SeminormedAddCommGroup W₃
    f₁ : NormedAddGroupHom V₁ W₁
    f₂ : NormedAddGroupHom V₂ W₂
    f₃ : NormedAddGroupHom V₃ W₃
    φ : NormedAddGroupHom V₁ V₂
    ψ : NormedAddGroupHom W₁ W₂
    φ' : NormedAddGroupHom V₂ V₃
    ψ' : NormedAddGroupHom W₂ W₃
    hf : Eq (ψ.comp f₁) (f₂.comp φ)
    hf' : Eq (ψ'.comp f₂) (f₃.comp φ')
    ⊢ Eq ((ψ'.comp ψ).comp f₁) (f₃.comp (φ'.comp φ))
  -/
  rw [comp_assoc, hf, ← comp_assoc, hf', comp_assoc]
  /-
    🎉 no goals
  -/


theorem map_comp_map (hf : ψ.comp f₁ = f₂.comp φ) (hg : ψ.comp g₁ = g₂.comp φ)
    (hf' : ψ'.comp f₂ = f₃.comp φ') (hg' : ψ'.comp g₂ = g₃.comp φ') :
    (map φ' ψ' hf' hg').comp (map φ ψ hf hg) =
      map (φ'.comp φ) (ψ'.comp ψ) (comm_sq₂ hf hf') (comm_sq₂ hg hg') := by
  /-
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝⁵ : SeminormedAddCommGroup V₁
    inst✝⁴ : SeminormedAddCommGroup V₂
    inst✝³ : SeminormedAddCommGroup V₃
    W₁ : Type u_6
    W₂ : Type u_7
    W₃ : Type u_8
    inst✝² : SeminormedAddCommGroup W₁
    inst✝¹ : SeminormedAddCommGroup W₂
    inst✝ : SeminormedAddCommGroup W₃
    f₁ g₁ : NormedAddGroupHom V₁ W₁
    f₂ g₂ : NormedAddGroupHom V₂ W₂
    f₃ g₃ : NormedAddGroupHom V₃ W₃
    φ : NormedAddGroupHom V₁ V₂
    ψ : NormedAddGroupHom W₁ W₂
    φ' : NormedAddGroupHom V₂ V₃
    ψ' : NormedAddGroupHom W₂ W₃
    hf : Eq (ψ.comp f₁) (f₂.comp φ)
    hg : Eq (ψ.comp g₁) (g₂.comp φ)
    hf' : Eq (ψ'.comp f₂) (f₃.comp φ')
    hg' : Eq (ψ'.comp g₂) (g₃.comp φ')
    ⊢ Eq ((NormedAddGroupHom.Equalizer.map φ' ψ' hf' hg').comp (NormedAddGroupHom. …
  -/
  ext
  /-
    case H.a
    V₁ : Type u_3
    V₂ : Type u_4
    V₃ : Type u_5
    inst✝⁵ : SeminormedAddCommGroup V₁
    inst✝⁴ : SeminormedAddCommGroup V₂
    inst✝³ : SeminormedAddCommGroup V₃
    W₁ : Type u_6
    W₂ : Type u_7
    W₃ : Type u_8
    inst✝² : SeminormedAddCommGroup W₁
    inst✝¹ : SeminormedAddCommGroup W₂
    inst✝ : SeminormedAddCommGroup W₃
    f₁ g₁ : NormedAddGroupHom V₁ W₁
    f₂ g₂ : NormedAddGroupHom V₂ W₂
    f₃ g₃ : NormedAddGroupHom V₃ W₃
    φ : NormedAddGroupHom V₁ V₂
    ψ : NormedAddGroupHom W₁ W₂
    φ' : NormedAddGroupHom V₂ V₃
    ψ' : NormedAddGroupHom W₂ W₃
    hf : Eq (ψ.comp f₁) (f₂.comp φ)
    hg : Eq (ψ.comp g₁) (g₂.comp φ)
    hf' : Eq (ψ'.comp f₂) (f₃.comp φ')
    hg' : Eq (ψ'.comp g₂) (g₃.comp φ')
    x✝ : Subtype fun x => Membership.mem (f₁.equalizer g₁) x
    ⊢ Eq ↑(((NormedAddGroupHom.Equalizer.map φ' ψ' hf' hg').comp (NormedAddGroupHo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ι_normNoninc : (ι f g).NormNoninc := fun _v => le_rfl


/-- The lifting of a norm nonincreasing morphism is norm nonincreasing. -/
theorem lift_normNoninc (φ : NormedAddGroupHom V₁ V) (h : f.comp φ = g.comp φ) (hφ : φ.NormNoninc) :
    (lift φ h).NormNoninc :=
  hφ


/-- If `φ` satisfies `‖φ‖ ≤ C`, then the same is true for the lifted morphism. -/
theorem norm_lift_le (φ : NormedAddGroupHom V₁ V) (h : f.comp φ = g.comp φ) (C : ℝ) (hφ : ‖φ‖ ≤ C) :
    ‖lift φ h‖ ≤ C :=
  hφ


theorem map_normNoninc (hf : ψ.comp f₁ = f₂.comp φ) (hg : ψ.comp g₁ = g₂.comp φ)
    (hφ : φ.NormNoninc) : (map φ ψ hf hg).NormNoninc :=
  lift_normNoninc _ _ <| hφ.comp ι_normNoninc


theorem norm_map_le (hf : ψ.comp f₁ = f₂.comp φ) (hg : ψ.comp g₁ = g₂.comp φ) (C : ℝ)
    (hφ : ‖φ.comp (ι f₁ g₁)‖ ≤ C) : ‖map φ ψ hf hg‖ ≤ C :=
  norm_lift_le _ _ _ hφ


