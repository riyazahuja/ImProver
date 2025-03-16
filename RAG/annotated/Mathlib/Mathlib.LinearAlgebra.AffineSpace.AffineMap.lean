/-- An `AffineMap k P1 P2` (notation: `P1 →ᵃ[k] P2`) is a map from `P1` to `P2` that
induces a corresponding linear map from `V1` to `V2`. -/
structure AffineMap (k : Type*) {V1 : Type*} (P1 : Type*) {V2 : Type*} (P2 : Type*) [Ring k]
  [AddCommGroup V1] [Module k V1] [AffineSpace V1 P1] [AddCommGroup V2] [Module k V2]
  [AffineSpace V2 P2] where
  toFun : P1 → P2
  linear : V1 →ₗ[k] V2
  map_vadd' : ∀ (p : P1) (v : V1), toFun (v +ᵥ p) = linear v +ᵥ toFun p


/-- An `AffineMap k P1 P2` (notation: `P1 →ᵃ[k] P2`) is a map from `P1` to `P2` that
induces a corresponding linear map from `V1` to `V2`. -/
notation:25 P1 " →ᵃ[" k:25 "] " P2:0 => AffineMap k P1 P2


instance AffineMap.instFunLike (k : Type*) {V1 : Type*} (P1 : Type*) {V2 : Type*} (P2 : Type*)
    [Ring k] [AddCommGroup V1] [Module k V1] [AffineSpace V1 P1] [AddCommGroup V2] [Module k V2]
    [AffineSpace V2 P2] : FunLike (P1 →ᵃ[k] P2) P1 P2 where
  coe := AffineMap.toFun
  coe_injective' := fun ⟨f, f_linear, f_add⟩ ⟨g, g_linear, g_add⟩ => fun (h : f = g) => by
    /-
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      x✝¹ x✝ : AffineMap k P1 P2
      f : P1 → P2
      f_linear : LinearMap (RingHom.id k) V1 V2
      f_add : ∀ (p : P1) (v : V1), Eq (f (HVAdd.hVAdd v p)) (HVAdd.hVAdd (f_linear v …
      g : P1 → P2
      g_linear : LinearMap (RingHom.id k) V1 V2
      g_add : ∀ (p : P1) (v : V1), Eq (g (HVAdd.hVAdd v p)) (HVAdd.hVAdd (g_linear v …
      h : Eq f g
      ⊢ Eq { toFun := f, linear := f_linear, map_vadd' := f_add } { toFun := g, line …
    -/
    cases' (AddTorsor.nonempty : Nonempty P1) with p
    /-
      case intro
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      x✝¹ x✝ : AffineMap k P1 P2
      f : P1 → P2
      f_linear : LinearMap (RingHom.id k) V1 V2
      f_add : ∀ (p : P1) (v : V1), Eq (f (HVAdd.hVAdd v p)) (HVAdd.hVAdd (f_linear v …
      g : P1 → P2
      g_linear : LinearMap (RingHom.id k) V1 V2
      g_add : ∀ (p : P1) (v : V1), Eq (g (HVAdd.hVAdd v p)) (HVAdd.hVAdd (g_linear v …
      h : Eq f g
      p : P1
      ⊢ Eq { toFun := f, linear := f_linear, map_vadd' := f_add } { toFun := g, line …
    -/
    congr with v
    /-
      case intro.e_linear.h
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      x✝¹ x✝ : AffineMap k P1 P2
      f : P1 → P2
      f_linear : LinearMap (RingHom.id k) V1 V2
      f_add : ∀ (p : P1) (v : V1), Eq (f (HVAdd.hVAdd v p)) (HVAdd.hVAdd (f_linear v …
      g : P1 → P2
      g_linear : LinearMap (RingHom.id k) V1 V2
      g_add : ∀ (p : P1) (v : V1), Eq (g (HVAdd.hVAdd v p)) (HVAdd.hVAdd (g_linear v …
      h : Eq f g
      p : P1
      v : V1
      ⊢ Eq (f_linear v) (g_linear v)
    -/
    apply vadd_right_cancel (f p)
    /-
      case intro.e_linear.h
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      x✝¹ x✝ : AffineMap k P1 P2
      f : P1 → P2
      f_linear : LinearMap (RingHom.id k) V1 V2
      f_add : ∀ (p : P1) (v : V1), Eq (f (HVAdd.hVAdd v p)) (HVAdd.hVAdd (f_linear v …
      g : P1 → P2
      g_linear : LinearMap (RingHom.id k) V1 V2
      g_add : ∀ (p : P1) (v : V1), Eq (g (HVAdd.hVAdd v p)) (HVAdd.hVAdd (g_linear v …
      h : Eq f g
      p : P1
      v : V1
      ⊢ Eq (HVAdd.hVAdd (f_linear v) (f p)) (HVAdd.hVAdd (g_linear v) (f p))
    -/
    rw [← f_add, h, ← g_add]
    /-
      🎉 no goals
    -/


/-- Reinterpret a linear map as an affine map. -/
def toAffineMap : V₁ →ᵃ[k] V₂ where
  toFun := f
  linear := f
  map_vadd' p v := f.map_add v p


@[simp]
theorem coe_toAffineMap : ⇑f.toAffineMap = f :=
  rfl


@[simp]
theorem toAffineMap_linear : f.toAffineMap.linear = f :=
  rfl


/-- Constructing an affine map and coercing back to a function
produces the same map. -/
@[simp]
theorem coe_mk (f : P1 → P2) (linear add) : ((mk f linear add : P1 →ᵃ[k] P2) : P1 → P2) = f :=
  rfl


/-- `toFun` is the same as the result of coercing to a function. -/
@[simp]
theorem toFun_eq_coe (f : P1 →ᵃ[k] P2) : f.toFun = ⇑f :=
  rfl


/-- An affine map on the result of adding a vector to a point produces
the same result as the linear map applied to that vector, added to the
affine map applied to that point. -/
@[simp]
theorem map_vadd (f : P1 →ᵃ[k] P2) (p : P1) (v : V1) : f (v +ᵥ p) = f.linear v +ᵥ f p :=
  f.map_vadd' p v


/-- The linear map on the result of subtracting two points is the
result of subtracting the result of the affine map on those two
points. -/
@[simp]
theorem linearMap_vsub (f : P1 →ᵃ[k] P2) (p1 p2 : P1) : f.linear (p1 -ᵥ p2) = f p1 -ᵥ f p2 := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    p1 p2 : P1
    ⊢ Eq (f.linear (VSub.vsub p1 p2)) (VSub.vsub (f p1) (f p2))
  -/
  conv_rhs => rw [← vsub_vadd p1 p2, map_vadd, vadd_vsub]
  /-
    🎉 no goals
  -/


/-- Two affine maps are equal if they coerce to the same function. -/
@[ext]
theorem ext {f g : P1 →ᵃ[k] P2} (h : ∀ p, f p = g p) : f = g :=
  DFunLike.ext _ _ h


theorem coeFn_injective : @Function.Injective (P1 →ᵃ[k] P2) (P1 → P2) (⇑) :=
  DFunLike.coe_injective


protected theorem congr_arg (f : P1 →ᵃ[k] P2) {x y : P1} (h : x = y) : f x = f y :=
  congr_arg _ h


protected theorem congr_fun {f g : P1 →ᵃ[k] P2} (h : f = g) (x : P1) : f x = g x :=
  h ▸ rfl


/-- Two affine maps are equal if they have equal linear maps and are equal at some point. -/
theorem ext_linear {f g : P1 →ᵃ[k] P2} (h₁ : f.linear = g.linear) {p : P1} (h₂ : f p = g p) :
    f = g := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f g : AffineMap k P1 P2
    h₁ : Eq f.linear g.linear
    p : P1
    h₂ : Eq (f p) (g p)
    ⊢ Eq f g
  -/
  ext q
  /-
    case h
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f g : AffineMap k P1 P2
    h₁ : Eq f.linear g.linear
    p : P1
    h₂ : Eq (f p) (g p)
    q : P1
    ⊢ Eq (f q) (g q)
  -/
  have hgl : g.linear (q -ᵥ p) = toFun g ((q -ᵥ p) +ᵥ q) -ᵥ toFun g q := by simp
  /-
    case h
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f g : AffineMap k P1 P2
    h₁ : Eq f.linear g.linear
    p : P1
    h₂ : Eq (f p) (g p)
    q : P1
    hgl : Eq (g.linear (VSub.vsub q p)) (VSub.vsub (g.toFun (HVAdd.hVAdd (VSub.vsu …
    ⊢ Eq (f q) (g q)
  -/
  have := f.map_vadd' q (q -ᵥ p)
  /-
    case h
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f g : AffineMap k P1 P2
    h₁ : Eq f.linear g.linear
    p : P1
    h₂ : Eq (f p) (g p)
    q : P1
    hgl : Eq (g.linear (VSub.vsub q p)) (VSub.vsub (g.toFun (HVAdd.hVAdd (VSub.vsu …
    this : Eq (f.toFun (HVAdd.hVAdd (VSub.vsub q p) q)) (HVAdd.hVAdd (f.linear (VS …
    ⊢ Eq (f q) (g q)
  -/
  rw [h₁, hgl, toFun_eq_coe, map_vadd, linearMap_vsub, h₂] at this
  /-
    case h
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f g : AffineMap k P1 P2
    h₁ : Eq f.linear g.linear
    p : P1
    h₂ : Eq (f p) (g p)
    q : P1
    hgl : Eq (g.linear (VSub.vsub q p)) (VSub.vsub (g.toFun (HVAdd.hVAdd (VSub.vsu …
    this : Eq (HVAdd.hVAdd (VSub.vsub (f q) (g p)) (f q)) (HVAdd.hVAdd (VSub.vsub  …
    ⊢ Eq (f q) (g q)
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- Two affine maps are equal if they have equal linear maps and are equal at some point. -/
theorem ext_linear_iff {f g : P1 →ᵃ[k] P2} : f = g ↔ (f.linear = g.linear) ∧ (∃ p, f p = g p) :=
                             /-
                               k : Type u_1
                               V1 : Type u_2
                               P1 : Type u_3
                               V2 : Type u_4
                               P2 : Type u_5
                               inst✝⁶ : Ring k
                               inst✝⁵ : AddCommGroup V1
                               inst✝⁴ : Module k V1
                               inst✝³ : AddTorsor V1 P1
                               inst✝² : AddCommGroup V2
                               inst✝¹ : Module k V2
                               inst✝ : AddTorsor V2 P2
                               f g : AffineMap k P1 P2
                               h : Eq f g
                               ⊢ P1
                             -/
                                         /-
                                           🎉 no goals
                                         -/
  ⟨fun h ↦ ⟨congrArg _ h, by inhabit P1; exact default, by rw [h]⟩,
                                                           /-
                                                             🎉 no goals
                                                           -/
  fun h ↦ Exists.casesOn h.2 fun _ hp ↦ ext_linear h.1 hp⟩


/-- The constant function as an `AffineMap`. -/
def const (p : P2) : P1 →ᵃ[k] P2 where
  toFun := Function.const P1 p
  linear := 0
  map_vadd' _ _ :=
    letI : AddAction V2 P2 := inferInstance
       /-
         k : Type u_1
         V1 : Type u_2
         P1 : Type u_3
         V2 : Type u_4
         P2 : Type u_5
         V3 : Type u_6
         P3 : Type u_7
         V4 : Type u_8
         P4 : Type u_9
         inst✝¹² : Ring k
         inst✝¹¹ : AddCommGroup V1
         inst✝¹⁰ : Module k V1
         inst✝⁹ : AddTorsor V1 P1
         inst✝⁸ : AddCommGroup V2
         inst✝⁷ : Module k V2
         inst✝⁶ : AddTorsor V2 P2
         inst✝⁵ : AddCommGroup V3
         inst✝⁴ : Module k V3
         inst✝³ : AddTorsor V3 P3
         inst✝² : AddCommGroup V4
         inst✝¹ : Module k V4
         inst✝ : AddTorsor V4 P4
         p : P2
         x✝¹ : P1
         x✝ : V1
         this : AddAction V2 P2 := inferInstance
         ⊢ Eq (Function.const P1 p (HVAdd.hVAdd x✝ x✝¹)) (HVAdd.hVAdd (0 x✝) (Function. …
       -/
    by simp
       /-
         🎉 no goals
       -/


@[simp]
theorem coe_const (p : P2) : ⇑(const k P1 p) = Function.const P1 p :=
  rfl


@[simp]
theorem const_apply (p : P2) (q : P1) : (const k P1 p) q = p := rfl


@[simp]
theorem const_linear (p : P2) : (const k P1 p).linear = 0 :=
  rfl


theorem linear_eq_zero_iff_exists_const (f : P1 →ᵃ[k] P2) :
    f.linear = 0 ↔ ∃ q, f = const k P1 q := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    ⊢ Iff (Eq f.linear 0) (Exists fun q => Eq f (AffineMap.const k P1 q))
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      f : AffineMap k P1 P2
      h : Eq f.linear 0
      ⊢ Exists fun q => Eq f (AffineMap.const k P1 q)
    -/
  · use f (Classical.arbitrary P1)
    /-
      case h
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      f : AffineMap k P1 P2
      h : Eq f.linear 0
      ⊢ Eq f (AffineMap.const k P1 (f (Classical.arbitrary P1)))
    -/
    ext
    rw [coe_const, Function.const_apply, ← @vsub_eq_zero_iff_eq V2, ← f.linearMap_vsub, h,
      LinearMap.zero_apply]
    /-
      case refine_2
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      f : AffineMap k P1 P2
      h : Exists fun q => Eq f (AffineMap.const k P1 q)
      ⊢ Eq f.linear 0
    -/
  · rcases h with ⟨q, rfl⟩
    /-
      case refine_2.intro
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      q : P2
      ⊢ Eq (AffineMap.const k P1 q).linear 0
    -/
    exact const_linear k P1 q
    /-
      🎉 no goals
    -/


instance nonempty : Nonempty (P1 →ᵃ[k] P2) :=
  (AddTorsor.nonempty : Nonempty P2).map <| const k P1


/-- Construct an affine map by verifying the relation between the map and its linear part at one
base point. Namely, this function takes a map `f : P₁ → P₂`, a linear map `f' : V₁ →ₗ[k] V₂`, and
a point `p` such that for any other point `p'` we have `f p' = f' (p' -ᵥ p) +ᵥ f p`. -/
def mk' (f : P1 → P2) (f' : V1 →ₗ[k] V2) (p : P1) (h : ∀ p' : P1, f p' = f' (p' -ᵥ p) +ᵥ f p) :
    P1 →ᵃ[k] P2 where
  toFun := f
  linear := f'
                       /-
                         k : Type u_1
                         V1 : Type u_2
                         P1 : Type u_3
                         V2 : Type u_4
                         P2 : Type u_5
                         V3 : Type u_6
                         P3 : Type u_7
                         V4 : Type u_8
                         P4 : Type u_9
                         inst✝¹² : Ring k
                         inst✝¹¹ : AddCommGroup V1
                         inst✝¹⁰ : Module k V1
                         inst✝⁹ : AddTorsor V1 P1
                         inst✝⁸ : AddCommGroup V2
                         inst✝⁷ : Module k V2
                         inst✝⁶ : AddTorsor V2 P2
                         inst✝⁵ : AddCommGroup V3
                         inst✝⁴ : Module k V3
                         inst✝³ : AddTorsor V3 P3
                         inst✝² : AddCommGroup V4
                         inst✝¹ : Module k V4
                         inst✝ : AddTorsor V4 P4
                         f : P1 → P2
                         f' : LinearMap (RingHom.id k) V1 V2
                         p : P1
                         h : ∀ (p' : P1), Eq (f p') (HVAdd.hVAdd (f' (VSub.vsub p' p)) (f p))
                         p' : P1
                         v : V1
                         ⊢ Eq (f (HVAdd.hVAdd v p')) (HVAdd.hVAdd (f' v) (f p'))
                       -/
  map_vadd' p' v := by rw [h, h p', vadd_vsub_assoc, f'.map_add, vadd_vadd]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem coe_mk' (f : P1 → P2) (f' : V1 →ₗ[k] V2) (p h) : ⇑(mk' f f' p h) = f :=
  rfl


@[simp]
theorem mk'_linear (f : P1 → P2) (f' : V1 →ₗ[k] V2) (p h) : (mk' f f' p h).linear = f' :=
  rfl


/-- The space of affine maps to a module inherits an `R`-action from the action on its codomain. -/
instance mulAction : MulAction R (P1 →ᵃ[k] V2) where
  -- Porting note: `map_vadd` is `simp`, but we still have to pass it explicitly
                                                   /-
                                                     k : Type u_1
                                                     V1 : Type u_2
                                                     P1 : Type u_3
                                                     V2 : Type u_4
                                                     P2 : Type u_5
                                                     V3 : Type u_6
                                                     P3 : Type u_7
                                                     V4 : Type u_8
                                                     P4 : Type u_9
                                                     inst✝¹⁵ : Ring k
                                                     inst✝¹⁴ : AddCommGroup V1
                                                     inst✝¹³ : Module k V1
                                                     inst✝¹² : AddTorsor V1 P1
                                                     inst✝¹¹ : AddCommGroup V2
                                                     inst✝¹⁰ : Module k V2
                                                     inst✝⁹ : AddTorsor V2 P2
                                                     inst✝⁸ : AddCommGroup V3
                                                     inst✝⁷ : Module k V3
                                                     inst✝⁶ : AddTorsor V3 P3
                                                     inst✝⁵ : AddCommGroup V4
                                                     inst✝⁴ : Module k V4
                                                     inst✝³ : AddTorsor V4 P4
                                                     R : Type u_10
                                                     inst✝² : Monoid R
                                                     inst✝¹ : DistribMulAction R V2
                                                     inst✝ : SMulCommClass k R V2
                                                     c : R
                                                     f : AffineMap k P1 V2
                                                     p : P1
                                                     v : V1
                                                     ⊢ Eq (HSMul.hSMul c (⇑f) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((HSMul.hSMul c f.lin …
                                                   -/
  smul c f := ⟨c • ⇑f, c • f.linear, fun p v => by simp [smul_add, map_vadd f]⟩
                                                   /-
                                                     🎉 no goals
                                                   -/
  one_smul _ := ext fun _ => one_smul _ _
  mul_smul _ _ _ := ext fun _ => mul_smul _ _ _


@[simp, norm_cast]
theorem coe_smul (c : R) (f : P1 →ᵃ[k] V2) : ⇑(c • f) = c • ⇑f :=
  rfl


@[simp]
theorem smul_linear (t : R) (f : P1 →ᵃ[k] V2) : (t • f).linear = t • f.linear :=
  rfl


instance isCentralScalar [DistribMulAction Rᵐᵒᵖ V2] [IsCentralScalar R V2] :
  IsCentralScalar R (P1 →ᵃ[k] V2) where
    op_smul_eq_smul _r _x := ext fun _ => op_smul_eq_smul _ _


instance : Zero (P1 →ᵃ[k] V2) where zero := ⟨0, 0, fun _ _ => (zero_vadd _ _).symm⟩


instance : Add (P1 →ᵃ[k] V2) where
                                                        /-
                                                          k : Type u_1
                                                          V1 : Type u_2
                                                          P1 : Type u_3
                                                          V2 : Type u_4
                                                          P2 : Type u_5
                                                          V3 : Type u_6
                                                          P3 : Type u_7
                                                          V4 : Type u_8
                                                          P4 : Type u_9
                                                          inst✝¹² : Ring k
                                                          inst✝¹¹ : AddCommGroup V1
                                                          inst✝¹⁰ : Module k V1
                                                          inst✝⁹ : AddTorsor V1 P1
                                                          inst✝⁸ : AddCommGroup V2
                                                          inst✝⁷ : Module k V2
                                                          inst✝⁶ : AddTorsor V2 P2
                                                          inst✝⁵ : AddCommGroup V3
                                                          inst✝⁴ : Module k V3
                                                          inst✝³ : AddTorsor V3 P3
                                                          inst✝² : AddCommGroup V4
                                                          inst✝¹ : Module k V4
                                                          inst✝ : AddTorsor V4 P4
                                                          f g : AffineMap k P1 V2
                                                          p : P1
                                                          v : V1
                                                          ⊢ Eq (HAdd.hAdd (⇑f) (⇑g) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((HAdd.hAdd f.linear …
                                                        -/
  add f g := ⟨f + g, f.linear + g.linear, fun p v => by simp [add_add_add_comm]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


instance : Sub (P1 →ᵃ[k] V2) where
                                                        /-
                                                          k : Type u_1
                                                          V1 : Type u_2
                                                          P1 : Type u_3
                                                          V2 : Type u_4
                                                          P2 : Type u_5
                                                          V3 : Type u_6
                                                          P3 : Type u_7
                                                          V4 : Type u_8
                                                          P4 : Type u_9
                                                          inst✝¹² : Ring k
                                                          inst✝¹¹ : AddCommGroup V1
                                                          inst✝¹⁰ : Module k V1
                                                          inst✝⁹ : AddTorsor V1 P1
                                                          inst✝⁸ : AddCommGroup V2
                                                          inst✝⁷ : Module k V2
                                                          inst✝⁶ : AddTorsor V2 P2
                                                          inst✝⁵ : AddCommGroup V3
                                                          inst✝⁴ : Module k V3
                                                          inst✝³ : AddTorsor V3 P3
                                                          inst✝² : AddCommGroup V4
                                                          inst✝¹ : Module k V4
                                                          inst✝ : AddTorsor V4 P4
                                                          f g : AffineMap k P1 V2
                                                          p : P1
                                                          v : V1
                                                          ⊢ Eq (HSub.hSub (⇑f) (⇑g) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((HSub.hSub f.linear …
                                                        -/
  sub f g := ⟨f - g, f.linear - g.linear, fun p v => by simp [sub_add_sub_comm]⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


instance : Neg (P1 →ᵃ[k] V2) where
                                         /-
                                           k : Type u_1
                                           V1 : Type u_2
                                           P1 : Type u_3
                                           V2 : Type u_4
                                           P2 : Type u_5
                                           V3 : Type u_6
                                           P3 : Type u_7
                                           V4 : Type u_8
                                           P4 : Type u_9
                                           inst✝¹² : Ring k
                                           inst✝¹¹ : AddCommGroup V1
                                           inst✝¹⁰ : Module k V1
                                           inst✝⁹ : AddTorsor V1 P1
                                           inst✝⁸ : AddCommGroup V2
                                           inst✝⁷ : Module k V2
                                           inst✝⁶ : AddTorsor V2 P2
                                           inst✝⁵ : AddCommGroup V3
                                           inst✝⁴ : Module k V3
                                           inst✝³ : AddTorsor V3 P3
                                           inst✝² : AddCommGroup V4
                                           inst✝¹ : Module k V4
                                           inst✝ : AddTorsor V4 P4
                                           f : AffineMap k P1 V2
                                           p : P1
                                           v : V1
                                           ⊢ Eq (Neg.neg (⇑f) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((Neg.neg f.linear) v) (Neg …
                                         -/
  neg f := ⟨-f, -f.linear, fun p v => by simp [add_comm, map_vadd f]⟩
                                         /-
                                           🎉 no goals
                                         -/


@[simp, norm_cast]
theorem coe_zero : ⇑(0 : P1 →ᵃ[k] V2) = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_add (f g : P1 →ᵃ[k] V2) : ⇑(f + g) = f + g :=
  rfl


@[simp, norm_cast]
theorem coe_neg (f : P1 →ᵃ[k] V2) : ⇑(-f) = -f :=
  rfl


@[simp, norm_cast]
theorem coe_sub (f g : P1 →ᵃ[k] V2) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem zero_linear : (0 : P1 →ᵃ[k] V2).linear = 0 :=
  rfl


@[simp]
theorem add_linear (f g : P1 →ᵃ[k] V2) : (f + g).linear = f.linear + g.linear :=
  rfl


@[simp]
theorem sub_linear (f g : P1 →ᵃ[k] V2) : (f - g).linear = f.linear - g.linear :=
  rfl


@[simp]
theorem neg_linear (f : P1 →ᵃ[k] V2) : (-f).linear = -f.linear :=
  rfl


/-- The set of affine maps to a vector space is an additive commutative group. -/
instance : AddCommGroup (P1 →ᵃ[k] V2) :=
  coeFn_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => coe_smul _ _)
    fun _ _ => coe_smul _ _


/-- The space of affine maps from `P1` to `P2` is an affine space over the space of affine maps
from `P1` to the vector space `V2` corresponding to `P2`. -/
instance : AffineSpace (P1 →ᵃ[k] V2) (P1 →ᵃ[k] P2) where
  vadd f g :=
    ⟨fun p => f p +ᵥ g p, f.linear + g.linear,
                    /-
                      k : Type u_1
                      V1 : Type u_2
                      P1 : Type u_3
                      V2 : Type u_4
                      P2 : Type u_5
                      V3 : Type u_6
                      P3 : Type u_7
                      V4 : Type u_8
                      P4 : Type u_9
                      inst✝¹² : Ring k
                      inst✝¹¹ : AddCommGroup V1
                      inst✝¹⁰ : Module k V1
                      inst✝⁹ : AddTorsor V1 P1
                      inst✝⁸ : AddCommGroup V2
                      inst✝⁷ : Module k V2
                      inst✝⁶ : AddTorsor V2 P2
                      inst✝⁵ : AddCommGroup V3
                      inst✝⁴ : Module k V3
                      inst✝³ : AddTorsor V3 P3
                      inst✝² : AddCommGroup V4
                      inst✝¹ : Module k V4
                      inst✝ : AddTorsor V4 P4
                      f : AffineMap k P1 V2
                      g : AffineMap k P1 P2
                      p : P1
                      v : V1
                      ⊢ Eq ((fun p => HVAdd.hVAdd (f p) (g p)) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((HAd …
                    -/
      fun p v => by simp [vadd_vadd, add_right_comm]⟩
                    /-
                      🎉 no goals
                    -/
  zero_vadd f := ext fun p => zero_vadd _ (f p)
  add_vadd f₁ f₂ f₃ := ext fun p => add_vadd (f₁ p) (f₂ p) (f₃ p)
  vsub f g :=
    ⟨fun p => f p -ᵥ g p, f.linear - g.linear, fun p v => by
      /-
        k : Type u_1
        V1 : Type u_2
        P1 : Type u_3
        V2 : Type u_4
        P2 : Type u_5
        V3 : Type u_6
        P3 : Type u_7
        V4 : Type u_8
        P4 : Type u_9
        inst✝¹² : Ring k
        inst✝¹¹ : AddCommGroup V1
        inst✝¹⁰ : Module k V1
        inst✝⁹ : AddTorsor V1 P1
        inst✝⁸ : AddCommGroup V2
        inst✝⁷ : Module k V2
        inst✝⁶ : AddTorsor V2 P2
        inst✝⁵ : AddCommGroup V3
        inst✝⁴ : Module k V3
        inst✝³ : AddTorsor V3 P3
        inst✝² : AddCommGroup V4
        inst✝¹ : Module k V4
        inst✝ : AddTorsor V4 P4
        f g : AffineMap k P1 P2
        p : P1
        v : V1
        ⊢ Eq ((fun p => VSub.vsub (f p) (g p)) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((HSub. …
      -/
      simp [vsub_vadd_eq_vsub_sub, vadd_vsub_assoc, add_sub, sub_add_eq_add_sub]⟩
      /-
        🎉 no goals
      -/
  vsub_vadd' f g := ext fun p => vsub_vadd (f p) (g p)
  vadd_vsub' f g := ext fun p => vadd_vsub (f p) (g p)


@[simp]
theorem vadd_apply (f : P1 →ᵃ[k] V2) (g : P1 →ᵃ[k] P2) (p : P1) : (f +ᵥ g) p = f p +ᵥ g p :=
  rfl


@[simp]
theorem vsub_apply (f g : P1 →ᵃ[k] P2) (p : P1) : (f -ᵥ g : P1 →ᵃ[k] V2) p = f p -ᵥ g p :=
  rfl


/-- `Prod.fst` as an `AffineMap`. -/
def fst : P1 × P2 →ᵃ[k] P1 where
  toFun := Prod.fst
  linear := LinearMap.fst k V1 V2
  map_vadd' _ _ := rfl


@[simp]
theorem coe_fst : ⇑(fst : P1 × P2 →ᵃ[k] P1) = Prod.fst :=
  rfl


@[simp]
theorem fst_linear : (fst : P1 × P2 →ᵃ[k] P1).linear = LinearMap.fst k V1 V2 :=
  rfl


/-- `Prod.snd` as an `AffineMap`. -/
def snd : P1 × P2 →ᵃ[k] P2 where
  toFun := Prod.snd
  linear := LinearMap.snd k V1 V2
  map_vadd' _ _ := rfl


@[simp]
theorem coe_snd : ⇑(snd : P1 × P2 →ᵃ[k] P2) = Prod.snd :=
  rfl


@[simp]
theorem snd_linear : (snd : P1 × P2 →ᵃ[k] P2).linear = LinearMap.snd k V1 V2 :=
  rfl


/-- Identity map as an affine map. -/
nonrec def id : P1 →ᵃ[k] P1 where
  toFun := id
  linear := LinearMap.id
  map_vadd' _ _ := rfl


/-- The identity affine map acts as the identity. -/
@[simp]
theorem coe_id : ⇑(id k P1) = _root_.id :=
  rfl


@[simp]
theorem id_linear : (id k P1).linear = LinearMap.id :=
  rfl


/-- The identity affine map acts as the identity. -/
theorem id_apply (p : P1) : id k P1 p = p :=
  rfl


instance : Inhabited (P1 →ᵃ[k] P1) :=
  ⟨id k P1⟩


/-- Composition of affine maps. -/
def comp (f : P2 →ᵃ[k] P3) (g : P1 →ᵃ[k] P2) : P1 →ᵃ[k] P3 where
  toFun := f ∘ g
  linear := f.linear.comp g.linear
  map_vadd' := by
    /-
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      V3 : Type u_6
      P3 : Type u_7
      V4 : Type u_8
      P4 : Type u_9
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : Module k V1
      inst✝⁹ : AddTorsor V1 P1
      inst✝⁸ : AddCommGroup V2
      inst✝⁷ : Module k V2
      inst✝⁶ : AddTorsor V2 P2
      inst✝⁵ : AddCommGroup V3
      inst✝⁴ : Module k V3
      inst✝³ : AddTorsor V3 P3
      inst✝² : AddCommGroup V4
      inst✝¹ : Module k V4
      inst✝ : AddTorsor V4 P4
      f : AffineMap k P2 P3
      g : AffineMap k P1 P2
      ⊢ ∀ (p : P1) (v : V1), Eq (Function.comp (⇑f) (⇑g) (HVAdd.hVAdd v p)) (HVAdd.h …
    -/
    intro p v
    /-
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      V3 : Type u_6
      P3 : Type u_7
      V4 : Type u_8
      P4 : Type u_9
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : Module k V1
      inst✝⁹ : AddTorsor V1 P1
      inst✝⁸ : AddCommGroup V2
      inst✝⁷ : Module k V2
      inst✝⁶ : AddTorsor V2 P2
      inst✝⁵ : AddCommGroup V3
      inst✝⁴ : Module k V3
      inst✝³ : AddTorsor V3 P3
      inst✝² : AddCommGroup V4
      inst✝¹ : Module k V4
      inst✝ : AddTorsor V4 P4
      f : AffineMap k P2 P3
      g : AffineMap k P1 P2
      p : P1
      v : V1
      ⊢ Eq (Function.comp (⇑f) (⇑g) (HVAdd.hVAdd v p)) (HVAdd.hVAdd ((f.linear.comp  …
    -/
    rw [Function.comp_apply, g.map_vadd, f.map_vadd]
    /-
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      V3 : Type u_6
      P3 : Type u_7
      V4 : Type u_8
      P4 : Type u_9
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : Module k V1
      inst✝⁹ : AddTorsor V1 P1
      inst✝⁸ : AddCommGroup V2
      inst✝⁷ : Module k V2
      inst✝⁶ : AddTorsor V2 P2
      inst✝⁵ : AddCommGroup V3
      inst✝⁴ : Module k V3
      inst✝³ : AddTorsor V3 P3
      inst✝² : AddCommGroup V4
      inst✝¹ : Module k V4
      inst✝ : AddTorsor V4 P4
      f : AffineMap k P2 P3
      g : AffineMap k P1 P2
      p : P1
      v : V1
      ⊢ Eq (HVAdd.hVAdd (f.linear (g.linear v)) (f (g p))) (HVAdd.hVAdd ((f.linear.c …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Composition of affine maps acts as applying the two functions. -/
@[simp]
theorem coe_comp (f : P2 →ᵃ[k] P3) (g : P1 →ᵃ[k] P2) : ⇑(f.comp g) = f ∘ g :=
  rfl


/-- Composition of affine maps acts as applying the two functions. -/
theorem comp_apply (f : P2 →ᵃ[k] P3) (g : P1 →ᵃ[k] P2) (p : P1) : f.comp g p = f (g p) :=
  rfl


@[simp]
theorem comp_id (f : P1 →ᵃ[k] P2) : f.comp (id k P1) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : P1 →ᵃ[k] P2) : (id k P2).comp f = f :=
  ext fun _ => rfl


theorem comp_assoc (f₃₄ : P3 →ᵃ[k] P4) (f₂₃ : P2 →ᵃ[k] P3) (f₁₂ : P1 →ᵃ[k] P2) :
    (f₃₄.comp f₂₃).comp f₁₂ = f₃₄.comp (f₂₃.comp f₁₂) :=
  rfl


instance : Monoid (P1 →ᵃ[k] P1) where
  one := id k P1
  mul := comp
  one_mul := id_comp
  mul_one := comp_id
  mul_assoc := comp_assoc


@[simp]
theorem coe_mul (f g : P1 →ᵃ[k] P1) : ⇑(f * g) = f ∘ g :=
  rfl


@[simp]
theorem coe_one : ⇑(1 : P1 →ᵃ[k] P1) = _root_.id :=
  rfl


/-- `AffineMap.linear` on endomorphisms is a `MonoidHom`. -/
@[simps]
def linearHom : (P1 →ᵃ[k] P1) →* V1 →ₗ[k] V1 where
  toFun := linear
  map_one' := rfl
  map_mul' _ _ := rfl


@[simp]
theorem linear_injective_iff (f : P1 →ᵃ[k] P2) :
    Function.Injective f.linear ↔ Function.Injective f := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    ⊢ Iff (Function.Injective ⇑f.linear) (Function.Injective ⇑f)
  -/
  obtain ⟨p⟩ := (inferInstance : Nonempty P1)
  have h : ⇑f.linear = (Equiv.vaddConst (f p)).symm ∘ f ∘ Equiv.vaddConst p := by
    ext v
    simp [f.map_vadd, vadd_vsub_assoc]
  /-
    case intro
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    p : P1
    h : Eq (⇑f.linear) (Function.comp (⇑(Equiv.vaddConst (f p)).symm) (Function.co …
    ⊢ Iff (Function.Injective ⇑f.linear) (Function.Injective ⇑f)
  -/
  rw [h, Equiv.comp_injective, Equiv.injective_comp]
  /-
    🎉 no goals
  -/


@[simp]
theorem linear_surjective_iff (f : P1 →ᵃ[k] P2) :
    Function.Surjective f.linear ↔ Function.Surjective f := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    ⊢ Iff (Function.Surjective ⇑f.linear) (Function.Surjective ⇑f)
  -/
  obtain ⟨p⟩ := (inferInstance : Nonempty P1)
  have h : ⇑f.linear = (Equiv.vaddConst (f p)).symm ∘ f ∘ Equiv.vaddConst p := by
    ext v
    simp [f.map_vadd, vadd_vsub_assoc]
  /-
    case intro
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    p : P1
    h : Eq (⇑f.linear) (Function.comp (⇑(Equiv.vaddConst (f p)).symm) (Function.co …
    ⊢ Iff (Function.Surjective ⇑f.linear) (Function.Surjective ⇑f)
  -/
  rw [h, Equiv.comp_surjective, Equiv.surjective_comp]
  /-
    🎉 no goals
  -/


@[simp]
theorem linear_bijective_iff (f : P1 →ᵃ[k] P2) :
    Function.Bijective f.linear ↔ Function.Bijective f :=
  and_congr f.linear_injective_iff f.linear_surjective_iff


theorem image_vsub_image {s t : Set P1} (f : P1 →ᵃ[k] P2) :
    f '' s -ᵥ f '' t = f.linear '' (s -ᵥ t) := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    s t : Set P1
    f : AffineMap k P1 P2
    ⊢ Eq (VSub.vsub (Set.image (⇑f) s) (Set.image (⇑f) t)) (Set.image (⇑f.linear)  …
  -/
  ext v
  -- Porting note: `simp` needs `Set.mem_vsub` to be an expression
  simp only [(Set.mem_vsub), Set.mem_image,
    exists_exists_and_eq_and, exists_and_left, ← f.linearMap_vsub]
  /-
    case h
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    s t : Set P1
    f : AffineMap k P1 P2
    v : V2
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Exists fun a_1 => And (Member …
  -/
  constructor
    /-
      case h.mp
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      s t : Set P1
      f : AffineMap k P1 P2
      v : V2
      ⊢ (Exists fun a => And (Membership.mem s a) (Exists fun a_1 => And (Membership …
    -/
  · rintro ⟨x, hx, y, hy, hv⟩
    /-
      case h.mp.intro.intro.intro.intro
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      s t : Set P1
      f : AffineMap k P1 P2
      v : V2
      x : P1
      hx : Membership.mem s x
      y : P1
      hy : Membership.mem t y
      hv : Eq (f.linear (VSub.vsub x y)) v
      ⊢ Exists fun x => And (Exists fun x_1 => And (Membership.mem s x_1) (Exists fu …
    -/
    exact ⟨x -ᵥ y, ⟨x, hx, y, hy, rfl⟩, hv⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      s t : Set P1
      f : AffineMap k P1 P2
      v : V2
      ⊢ (Exists fun x => And (Exists fun x_1 => And (Membership.mem s x_1) (Exists f …
    -/
  · rintro ⟨-, ⟨x, hx, y, hy, rfl⟩, rfl⟩
    /-
      case h.mpr.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V1 : Type u_2
      P1 : Type u_3
      V2 : Type u_4
      P2 : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V1
      inst✝⁴ : Module k V1
      inst✝³ : AddTorsor V1 P1
      inst✝² : AddCommGroup V2
      inst✝¹ : Module k V2
      inst✝ : AddTorsor V2 P2
      s t : Set P1
      f : AffineMap k P1 P2
      x : P1
      hx : Membership.mem s x
      y : P1
      hy : Membership.mem t y
      ⊢ Exists fun a => And (Membership.mem s a) (Exists fun a_1 => And (Membership. …
    -/
    exact ⟨x, hx, y, hy, rfl⟩
    /-
      🎉 no goals
    -/


/-- The affine map from `k` to `P1` sending `0` to `p₀` and `1` to `p₁`. -/
def lineMap (p₀ p₁ : P1) : k →ᵃ[k] P1 :=
  ((LinearMap.id : k →ₗ[k] k).smulRight (p₁ -ᵥ p₀)).toAffineMap +ᵥ const k k p₀


theorem coe_lineMap (p₀ p₁ : P1) : (lineMap p₀ p₁ : k → P1) = fun c => c • (p₁ -ᵥ p₀) +ᵥ p₀ :=
  rfl


theorem lineMap_apply (p₀ p₁ : P1) (c : k) : lineMap p₀ p₁ c = c • (p₁ -ᵥ p₀) +ᵥ p₀ :=
  rfl


theorem lineMap_apply_module' (p₀ p₁ : V1) (c : k) : lineMap p₀ p₁ c = c • (p₁ - p₀) + p₀ :=
  rfl


theorem lineMap_apply_module (p₀ p₁ : V1) (c : k) : lineMap p₀ p₁ c = (1 - c) • p₀ + c • p₁ := by
  /-
    k : Type u_1
    V1 : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V1
    inst✝ : Module k V1
    p₀ p₁ : V1
    c : k
    ⊢ Eq ((AffineMap.lineMap p₀ p₁) c) (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 c) p₀) …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  simp [lineMap_apply_module', smul_sub, sub_smul]; abel
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem lineMap_apply_ring' (a b c : k) : lineMap a b c = c * (b - a) + a :=
  rfl


theorem lineMap_apply_ring (a b c : k) : lineMap a b c = (1 - c) * a + c * b :=
  lineMap_apply_module a b c


theorem lineMap_vadd_apply (p : P1) (v : V1) (c : k) : lineMap p (v +ᵥ p) c = c • v +ᵥ p := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p : P1
    v : V1
    c : k
    ⊢ Eq ((AffineMap.lineMap p (HVAdd.hVAdd v p)) c) (HVAdd.hVAdd (HSMul.hSMul c v …
  -/
  rw [lineMap_apply, vadd_vsub]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_linear (p₀ p₁ : P1) :
    (lineMap p₀ p₁ : k →ᵃ[k] P1).linear = LinearMap.id.smulRight (p₁ -ᵥ p₀) :=
  add_zero _


theorem lineMap_same_apply (p : P1) (c : k) : lineMap p p c = p := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p : P1
    c : k
    ⊢ Eq ((AffineMap.lineMap p p) c) p
  -/
  simp [lineMap_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_same (p : P1) : lineMap p p = const k k p :=
  ext <| lineMap_same_apply p


@[simp]
theorem lineMap_apply_zero (p₀ p₁ : P1) : lineMap p₀ p₁ (0 : k) = p₀ := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    ⊢ Eq ((AffineMap.lineMap p₀ p₁) 0) p₀
  -/
  simp [lineMap_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_apply_one (p₀ p₁ : P1) : lineMap p₀ p₁ (1 : k) = p₁ := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    ⊢ Eq ((AffineMap.lineMap p₀ p₁) 1) p₁
  -/
  simp [lineMap_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_eq_lineMap_iff [NoZeroSMulDivisors k V1] {p₀ p₁ : P1} {c₁ c₂ : k} :
    lineMap p₀ p₁ c₁ = lineMap p₀ p₁ c₂ ↔ p₀ = p₁ ∨ c₁ = c₂ := by
  rw [lineMap_apply, lineMap_apply, ← @vsub_eq_zero_iff_eq V1, vadd_vsub_vadd_cancel_right, ←
    sub_smul, smul_eq_zero, sub_eq_zero, vsub_eq_zero_iff_eq, or_comm, eq_comm]


@[simp]
theorem lineMap_eq_left_iff [NoZeroSMulDivisors k V1] {p₀ p₁ : P1} {c : k} :
    lineMap p₀ p₁ c = p₀ ↔ p₀ = p₁ ∨ c = 0 := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V1
    inst✝² : Module k V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : NoZeroSMulDivisors k V1
    p₀ p₁ : P1
    c : k
    ⊢ Iff (Eq ((AffineMap.lineMap p₀ p₁) c) p₀) (Or (Eq p₀ p₁) (Eq c 0))
  -/
  rw [← @lineMap_eq_lineMap_iff k V1, lineMap_apply_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_eq_right_iff [NoZeroSMulDivisors k V1] {p₀ p₁ : P1} {c : k} :
    lineMap p₀ p₁ c = p₁ ↔ p₀ = p₁ ∨ c = 1 := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V1
    inst✝² : Module k V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : NoZeroSMulDivisors k V1
    p₀ p₁ : P1
    c : k
    ⊢ Iff (Eq ((AffineMap.lineMap p₀ p₁) c) p₁) (Or (Eq p₀ p₁) (Eq c 1))
  -/
  rw [← @lineMap_eq_lineMap_iff k V1, lineMap_apply_one]
  /-
    🎉 no goals
  -/


theorem lineMap_injective [NoZeroSMulDivisors k V1] {p₀ p₁ : P1} (h : p₀ ≠ p₁) :
    Function.Injective (lineMap p₀ p₁ : k → P1) := fun _c₁ _c₂ hc =>
  (lineMap_eq_lineMap_iff.mp hc).resolve_left h


@[simp]
theorem apply_lineMap (f : P1 →ᵃ[k] P2) (p₀ p₁ : P1) (c : k) :
    f (lineMap p₀ p₁ c) = lineMap (f p₀) (f p₁) c := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    V2 : Type u_4
    P2 : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V1
    inst✝⁴ : Module k V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : AddCommGroup V2
    inst✝¹ : Module k V2
    inst✝ : AddTorsor V2 P2
    f : AffineMap k P1 P2
    p₀ p₁ : P1
    c : k
    ⊢ Eq (f ((AffineMap.lineMap p₀ p₁) c)) ((AffineMap.lineMap (f p₀) (f p₁)) c)
  -/
  simp [lineMap_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_lineMap (f : P1 →ᵃ[k] P2) (p₀ p₁ : P1) :
    f.comp (lineMap p₀ p₁) = lineMap (f p₀) (f p₁) :=
  ext <| f.apply_lineMap p₀ p₁


@[simp]
theorem fst_lineMap (p₀ p₁ : P1 × P2) (c : k) : (lineMap p₀ p₁ c).1 = lineMap p₀.1 p₁.1 c :=
  fst.apply_lineMap p₀ p₁ c


@[simp]
theorem snd_lineMap (p₀ p₁ : P1 × P2) (c : k) : (lineMap p₀ p₁ c).2 = lineMap p₀.2 p₁.2 c :=
  snd.apply_lineMap p₀ p₁ c


theorem lineMap_symm (p₀ p₁ : P1) :
    lineMap p₀ p₁ = (lineMap p₁ p₀).comp (lineMap (1 : k) (0 : k)) := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    ⊢ Eq (AffineMap.lineMap p₀ p₁) ((AffineMap.lineMap p₁ p₀).comp (AffineMap.line …
  -/
  rw [comp_lineMap]
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    ⊢ Eq (AffineMap.lineMap p₀ p₁) (AffineMap.lineMap ((AffineMap.lineMap p₁ p₀) 1 …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem lineMap_apply_one_sub (p₀ p₁ : P1) (c : k) : lineMap p₀ p₁ (1 - c) = lineMap p₁ p₀ c := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c : k
    ⊢ Eq ((AffineMap.lineMap p₀ p₁) (HSub.hSub 1 c)) ((AffineMap.lineMap p₁ p₀) c)
  -/
  rw [lineMap_symm p₀, comp_apply]
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c : k
    ⊢ Eq ((AffineMap.lineMap p₁ p₀) ((AffineMap.lineMap 1 0) (HSub.hSub 1 c))) ((A …
  -/
  congr
  /-
    case h.e_6.h
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c : k
    ⊢ Eq ((AffineMap.lineMap 1 0) (HSub.hSub 1 c)) c
  -/
  simp [lineMap_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_vsub_left (p₀ p₁ : P1) (c : k) : lineMap p₀ p₁ c -ᵥ p₀ = c • (p₁ -ᵥ p₀) :=
  vadd_vsub _ _


@[simp]
theorem left_vsub_lineMap (p₀ p₁ : P1) (c : k) : p₀ -ᵥ lineMap p₀ p₁ c = c • (p₀ -ᵥ p₁) := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c : k
    ⊢ Eq (VSub.vsub p₀ ((AffineMap.lineMap p₀ p₁) c)) (HSMul.hSMul c (VSub.vsub p₀ …
  -/
  rw [← neg_vsub_eq_vsub_rev, lineMap_vsub_left, ← smul_neg, neg_vsub_eq_vsub_rev]
  /-
    🎉 no goals
  -/


@[simp]
theorem lineMap_vsub_right (p₀ p₁ : P1) (c : k) : lineMap p₀ p₁ c -ᵥ p₁ = (1 - c) • (p₀ -ᵥ p₁) := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c : k
    ⊢ Eq (VSub.vsub ((AffineMap.lineMap p₀ p₁) c) p₁) (HSMul.hSMul (HSub.hSub 1 c) …
  -/
  rw [← lineMap_apply_one_sub, lineMap_vsub_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem right_vsub_lineMap (p₀ p₁ : P1) (c : k) : p₁ -ᵥ lineMap p₀ p₁ c = (1 - c) • (p₁ -ᵥ p₀) := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c : k
    ⊢ Eq (VSub.vsub p₁ ((AffineMap.lineMap p₀ p₁) c)) (HSMul.hSMul (HSub.hSub 1 c) …
  -/
  rw [← lineMap_apply_one_sub, left_vsub_lineMap]
  /-
    🎉 no goals
  -/


theorem lineMap_vadd_lineMap (v₁ v₂ : V1) (p₁ p₂ : P1) (c : k) :
    lineMap v₁ v₂ c +ᵥ lineMap p₁ p₂ c = lineMap (v₁ +ᵥ p₁) (v₂ +ᵥ p₂) c :=
  ((fst : V1 × P1 →ᵃ[k] V1) +ᵥ (snd : V1 × P1 →ᵃ[k] P1)).apply_lineMap (v₁, p₁) (v₂, p₂) c


theorem lineMap_vsub_lineMap (p₁ p₂ p₃ p₄ : P1) (c : k) :
    lineMap p₁ p₂ c -ᵥ lineMap p₃ p₄ c = lineMap (p₁ -ᵥ p₃) (p₂ -ᵥ p₄) c :=
  ((fst : P1 × P1 →ᵃ[k] P1) -ᵥ (snd : P1 × P1 →ᵃ[k] P1)).apply_lineMap (_, _) (_, _) c


@[simp] lemma lineMap_lineMap_right (p₀ p₁ : P1) (c d : k) :
                                                                 /-
                                                                   k : Type u_1
                                                                   V1 : Type u_2
                                                                   P1 : Type u_3
                                                                   inst✝³ : Ring k
                                                                   inst✝² : AddCommGroup V1
                                                                   inst✝¹ : Module k V1
                                                                   inst✝ : AddTorsor V1 P1
                                                                   p₀ p₁ : P1
                                                                   c d : k
                                                                   ⊢ Eq ((AffineMap.lineMap p₀ ((AffineMap.lineMap p₀ p₁) c)) d) ((AffineMap.line …
                                                                 -/
    lineMap p₀ (lineMap p₀ p₁ c) d = lineMap p₀ p₁ (d * c) := by simp [lineMap_apply, mul_smul]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp] lemma lineMap_lineMap_left (p₀ p₁ : P1) (c d : k) :
    lineMap (lineMap p₀ p₁ c) p₁ d = lineMap p₀ p₁ (1 - (1 - d) * (1 - c)) := by
  /-
    k : Type u_1
    V1 : Type u_2
    P1 : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V1
    inst✝¹ : Module k V1
    inst✝ : AddTorsor V1 P1
    p₀ p₁ : P1
    c d : k
    ⊢ Eq ((AffineMap.lineMap ((AffineMap.lineMap p₀ p₁) c) p₁) d) ((AffineMap.line …
  -/
  simp_rw [lineMap_apply_one_sub, ← lineMap_apply_one_sub p₁, lineMap_lineMap_right]
  /-
    🎉 no goals
  -/


/-- Decomposition of an affine map in the special case when the point space and vector space
are the same. -/
theorem decomp (f : V1 →ᵃ[k] V2) : (f : V1 → V2) = ⇑f.linear + fun _ => f 0 := by
  /-
    k : Type u_1
    V1 : Type u_2
    V2 : Type u_4
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V1
    inst✝² : Module k V1
    inst✝¹ : AddCommGroup V2
    inst✝ : Module k V2
    f : AffineMap k V1 V2
    ⊢ Eq (⇑f) (HAdd.hAdd ⇑f.linear fun x => f 0)
  -/
  ext x
  calc
    f x = f.linear x +ᵥ f 0 := by rw [← f.map_vadd, vadd_eq_add, add_zero]
    _ = (f.linear + fun _ : V1 => f 0) x := rfl


/-- Decomposition of an affine map in the special case when the point space and vector space
are the same. -/
theorem decomp' (f : V1 →ᵃ[k] V2) : (f.linear : V1 → V2) = ⇑f - fun _ => f 0 := by
  /-
    k : Type u_1
    V1 : Type u_2
    V2 : Type u_4
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V1
    inst✝² : Module k V1
    inst✝¹ : AddCommGroup V2
    inst✝ : Module k V2
    f : AffineMap k V1 V2
    ⊢ Eq (⇑f.linear) (HSub.hSub ⇑f fun x => f 0)
  -/
  rw [decomp]
  /-
    k : Type u_1
    V1 : Type u_2
    V2 : Type u_4
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V1
    inst✝² : Module k V1
    inst✝¹ : AddCommGroup V2
    inst✝ : Module k V2
    f : AffineMap k V1 V2
    ⊢ Eq (⇑f.linear) (HSub.hSub (HAdd.hAdd ⇑f.linear fun x => f 0) fun x => HAdd.h …
  -/
  simp only [LinearMap.map_zero, Pi.add_apply, add_sub_cancel_right, zero_add]
  /-
    🎉 no goals
  -/


theorem image_uIcc {k : Type*} [LinearOrderedField k] (f : k →ᵃ[k] k) (a b : k) :
    f '' Set.uIcc a b = Set.uIcc (f a) (f b) := by
  have : ⇑f = (fun x => x + f 0) ∘ fun x => x * (f 1 - f 0) := by
    ext x
    change f x = x • (f 1 -ᵥ f 0) +ᵥ f 0
    rw [← f.linearMap_vsub, ← f.linear.map_smul, ← f.map_vadd]
    simp only [vsub_eq_sub, add_zero, mul_one, vadd_eq_add, sub_zero, smul_eq_mul]
  /-
    k : Type u_10
    inst✝ : LinearOrderedField k
    f : AffineMap k k k
    a b : k
    this : Eq (⇑f) (Function.comp (fun x => HAdd.hAdd x (f 0)) fun x => HMul.hMul  …
    ⊢ Eq (Set.image (⇑f) (Set.uIcc a b)) (Set.uIcc (f a) (f b))
  -/
  rw [this, Set.image_comp]
  /-
    k : Type u_10
    inst✝ : LinearOrderedField k
    f : AffineMap k k k
    a b : k
    this : Eq (⇑f) (Function.comp (fun x => HAdd.hAdd x (f 0)) fun x => HMul.hMul  …
    ⊢ Eq (Set.image (fun x => HAdd.hAdd x (f 0)) (Set.image (fun x => HMul.hMul x  …
  -/
  simp only [Set.image_add_const_uIcc, Set.image_mul_const_uIcc, Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- Evaluation at a point as an affine map. -/
def proj (i : ι) : (∀ i : ι, P i) →ᵃ[k] P i where
  toFun f := f i
  linear := @LinearMap.proj k ι _ V _ _ i
  map_vadd' _ _ := rfl


@[simp]
theorem proj_apply (i : ι) (f : ∀ i, P i) : @proj k _ ι V P _ _ _ i f = f i :=
  rfl


@[simp]
theorem proj_linear (i : ι) : (@proj k _ ι V P _ _ _ i).linear = @LinearMap.proj k ι _ V _ _ i :=
  rfl


theorem pi_lineMap_apply (f g : ∀ i, P i) (c : k) (i : ι) :
    lineMap f g c i = lineMap (f i) (g i) c :=
  (proj i : (∀ i, P i) →ᵃ[k] P i).apply_lineMap f g c


/-- The space of affine maps to a module inherits an `R`-action from the action on its codomain. -/
instance distribMulAction : DistribMulAction R (P1 →ᵃ[k] V2) where
  smul_add _ _ _ := ext fun _ => smul_add _ _ _
  smul_zero _ := ext fun _ => smul_zero _


/-- The space of affine maps taking values in an `R`-module is an `R`-module. -/
instance : Module R (P1 →ᵃ[k] V2) :=
  { AffineMap.distribMulAction with
    add_smul := fun _ _ _ => ext fun _ => add_smul _ _ _
    zero_smul := fun _ => ext fun _ => zero_smul _ _ }


/-- The space of affine maps between two modules is linearly equivalent to the product of the
domain with the space of linear maps, by taking the value of the affine map at `(0 : V1)` and the
linear part.

See note [bundled maps over different rings]-/
@[simps]
def toConstProdLinearMap : (V1 →ᵃ[k] V2) ≃ₗ[R] V2 × (V1 →ₗ[k] V2) where
  toFun f := ⟨f 0, f.linear⟩
  invFun p := p.2.toAffineMap + const k V1 p.1
  left_inv f := by
    /-
      R : Type u_1
      k : Type u_2
      V1 : Type u_3
      P1 : Type u_4
      V2 : Type u_5
      P2 : Type u_6
      V3 : Type u_7
      P3 : Type u_8
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : AddTorsor V1 P1
      inst✝⁹ : AddCommGroup V2
      inst✝⁸ : AddTorsor V2 P2
      inst✝⁷ : AddCommGroup V3
      inst✝⁶ : AddTorsor V3 P3
      inst✝⁵ : Module k V1
      inst✝⁴ : Module k V2
      inst✝³ : Module k V3
      inst✝² : Semiring R
      inst✝¹ : Module R V2
      inst✝ : SMulCommClass k R V2
      f : AffineMap k V1 V2
      ⊢ Eq ((fun p => HAdd.hAdd p.2.toAffineMap (AffineMap.const k V1 p.1)) ({ toFun …
    -/
    ext
    /-
      case h
      R : Type u_1
      k : Type u_2
      V1 : Type u_3
      P1 : Type u_4
      V2 : Type u_5
      P2 : Type u_6
      V3 : Type u_7
      P3 : Type u_8
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : AddTorsor V1 P1
      inst✝⁹ : AddCommGroup V2
      inst✝⁸ : AddTorsor V2 P2
      inst✝⁷ : AddCommGroup V3
      inst✝⁶ : AddTorsor V3 P3
      inst✝⁵ : Module k V1
      inst✝⁴ : Module k V2
      inst✝³ : Module k V3
      inst✝² : Semiring R
      inst✝¹ : Module R V2
      inst✝ : SMulCommClass k R V2
      f : AffineMap k V1 V2
      p✝ : V1
      ⊢ Eq (((fun p => HAdd.hAdd p.2.toAffineMap (AffineMap.const k V1 p.1)) ({ toFu …
    -/
    rw [f.decomp]
    /-
      case h
      R : Type u_1
      k : Type u_2
      V1 : Type u_3
      P1 : Type u_4
      V2 : Type u_5
      P2 : Type u_6
      V3 : Type u_7
      P3 : Type u_8
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : AddTorsor V1 P1
      inst✝⁹ : AddCommGroup V2
      inst✝⁸ : AddTorsor V2 P2
      inst✝⁷ : AddCommGroup V3
      inst✝⁶ : AddTorsor V3 P3
      inst✝⁵ : Module k V1
      inst✝⁴ : Module k V2
      inst✝³ : Module k V3
      inst✝² : Semiring R
      inst✝¹ : Module R V2
      inst✝ : SMulCommClass k R V2
      f : AffineMap k V1 V2
      p✝ : V1
      ⊢ Eq (((fun p => HAdd.hAdd p.2.toAffineMap (AffineMap.const k V1 p.1)) ({ toFu …
    -/
    simp [const_apply _ _]  -- Porting note: `simp` needs `_`s to use this lemma
                 /-
                   R : Type u_1
                   k : Type u_2
                   V1 : Type u_3
                   P1 : Type u_4
                   V2 : Type u_5
                   P2 : Type u_6
                   V3 : Type u_7
                   P3 : Type u_8
                   inst✝¹² : Ring k
                   inst✝¹¹ : AddCommGroup V1
                   inst✝¹⁰ : AddTorsor V1 P1
                   inst✝⁹ : AddCommGroup V2
                   inst✝⁸ : AddTorsor V2 P2
                   inst✝⁷ : AddCommGroup V3
                   inst✝⁶ : AddTorsor V3 P3
                   inst✝⁵ : Module k V1
                   inst✝⁴ : Module k V2
                   inst✝³ : Module k V3
                   inst✝² : Semiring R
                   inst✝¹ : Module R V2
                   inst✝ : SMulCommClass k R V2
                   ⊢ ∀ (x y : AffineMap k V1 V2), Eq ((fun f => { fst := f 0, snd := f.linear })  …
                 -/
    /-
      🎉 no goals
    -/
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    k : Type u_2
                    V1 : Type u_3
                    P1 : Type u_4
                    V2 : Type u_5
                    P2 : Type u_6
                    V3 : Type u_7
                    P3 : Type u_8
                    inst✝¹² : Ring k
                    inst✝¹¹ : AddCommGroup V1
                    inst✝¹⁰ : AddTorsor V1 P1
                    inst✝⁹ : AddCommGroup V2
                    inst✝⁸ : AddTorsor V2 P2
                    inst✝⁷ : AddCommGroup V3
                    inst✝⁶ : AddTorsor V3 P3
                    inst✝⁵ : Module k V1
                    inst✝⁴ : Module k V2
                    inst✝³ : Module k V3
                    inst✝² : Semiring R
                    inst✝¹ : Module R V2
                    inst✝ : SMulCommClass k R V2
                    ⊢ ∀ (m : R) (x : AffineMap k V1 V2), Eq ({ toFun := fun f => { fst := f 0, snd …
                  -/
  right_inv := by
                  /-
                    🎉 no goals
                  -/
    /-
      R : Type u_1
      k : Type u_2
      V1 : Type u_3
      P1 : Type u_4
      V2 : Type u_5
      P2 : Type u_6
      V3 : Type u_7
      P3 : Type u_8
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : AddTorsor V1 P1
      inst✝⁹ : AddCommGroup V2
      inst✝⁸ : AddTorsor V2 P2
      inst✝⁷ : AddCommGroup V3
      inst✝⁶ : AddTorsor V3 P3
      inst✝⁵ : Module k V1
      inst✝⁴ : Module k V2
      inst✝³ : Module k V3
      inst✝² : Semiring R
      inst✝¹ : Module R V2
      inst✝ : SMulCommClass k R V2
      ⊢ Function.RightInverse (fun p => HAdd.hAdd p.2.toAffineMap (AffineMap.const k …
    -/
    rintro ⟨v, f⟩
    /-
      case mk
      R : Type u_1
      k : Type u_2
      V1 : Type u_3
      P1 : Type u_4
      V2 : Type u_5
      P2 : Type u_6
      V3 : Type u_7
      P3 : Type u_8
      inst✝¹² : Ring k
      inst✝¹¹ : AddCommGroup V1
      inst✝¹⁰ : AddTorsor V1 P1
      inst✝⁹ : AddCommGroup V2
      inst✝⁸ : AddTorsor V2 P2
      inst✝⁷ : AddCommGroup V3
      inst✝⁶ : AddTorsor V3 P3
      inst✝⁵ : Module k V1
      inst✝⁴ : Module k V2
      inst✝³ : Module k V3
      inst✝² : Semiring R
      inst✝¹ : Module R V2
      inst✝ : SMulCommClass k R V2
      v : V2
      f : LinearMap (RingHom.id k) V1 V2
      ⊢ Eq ({ toFun := fun f => { fst := f 0, snd := f.linear }, map_add' := ⋯, map_ …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp [const_apply _ _, const_linear _ _]  -- Porting note: `simp` needs `_`s
            /-
              🎉 no goals
            -/
  map_add' := by simp
  map_smul' := by simp


/-- `pi` construction for affine maps. From a family of affine maps it produces an affine
map into a family of affine spaces.

This is the affine version of `LinearMap.pi`.
-/
def pi (f : (i : ι) → (P1 →ᵃ[k] φp i)) : P1 →ᵃ[k] ((i : ι) → φp i) where
  toFun m a := f a m
  linear := LinearMap.pi (fun a ↦ (f a).linear)
  map_vadd' _ _ := funext fun _ ↦ map_vadd _ _ _

--fp for when the image is a dependent AffineSpace φp i, fv for when the
--image is a Module φv i, f' for when the image isn't dependent.

@[simp]
theorem pi_apply (c : P1) (i : ι) : pi fp c i = fp i c :=
  rfl


theorem pi_comp (g : P3 →ᵃ[k] P1) : (pi fp).comp g = pi (fun i => (fp i).comp g) :=
  rfl


theorem pi_eq_zero : pi fv = 0 ↔ ∀ i, fv i = 0 := by
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : Module k V1
    ι : Type u_9
    φv : ι → Type u_10
    inst✝¹ : (i : ι) → AddCommGroup (φv i)
    inst✝ : (i : ι) → Module k (φv i)
    fv : (i : ι) → AffineMap k P1 (φv i)
    ⊢ Iff (Eq (AffineMap.pi fv) 0) (∀ (i : ι), Eq (fv i) 0)
  -/
  simp only [AffineMap.ext_iff, funext_iff, pi_apply]
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : Module k V1
    ι : Type u_9
    φv : ι → Type u_10
    inst✝¹ : (i : ι) → AddCommGroup (φv i)
    inst✝ : (i : ι) → Module k (φv i)
    fv : (i : ι) → AffineMap k P1 (φv i)
    ⊢ Iff (∀ (p : P1) (x : ι), Eq ((fv x) p) (0 p x)) (∀ (i : ι) (p : P1), Eq ((fv …
  -/
  exact forall_comm
  /-
    🎉 no goals
  -/


theorem pi_zero : pi (fun _ ↦ 0 : (i : ι) → P1 →ᵃ[k] φv i) = 0 := by
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V1
    inst✝³ : AddTorsor V1 P1
    inst✝² : Module k V1
    ι : Type u_9
    φv : ι → Type u_10
    inst✝¹ : (i : ι) → AddCommGroup (φv i)
    inst✝ : (i : ι) → Module k (φv i)
    ⊢ Eq (AffineMap.pi fun x => 0) 0
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


theorem proj_pi (i : ι) : (proj i).comp (pi fp) = fp i :=
  ext fun _ => rfl

/-- Two affine maps from a Pi-type of modules `(i : ι) → φv i` are equal if they are equal in their
  operation on `Pi.single` and at zero. Analogous to `LinearMap.pi_ext`. See also `pi_ext_nonempty`,
  which instead of agreement at zero requires `Nonempty ι`. -/
theorem pi_ext_zero (h : ∀ i x, f (Pi.single i x) = g (Pi.single i x)) (h₂ : f 0 = g 0) :
    f = g := by
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁷ : Ring k
    inst✝⁶ : AddCommGroup V2
    inst✝⁵ : AddTorsor V2 P2
    inst✝⁴ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝³ : (i : ι) → AddCommGroup (φv i)
    inst✝² : (i : ι) → Module k (φv i)
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
    h₂ : Eq (f 0) (g 0)
    ⊢ Eq f g
  -/
  apply ext_linear
    /-
      case h₁
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      ⊢ Eq f.linear g.linear
    -/
  · apply LinearMap.pi_ext
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      ⊢ ∀ (i : ι) (x : φv i), Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    intro i x
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      i : ι
      x : φv i
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    have s₁ := h i x
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      i : ι
      x : φv i
      s₁ : Eq (f (Pi.single i x)) (g (Pi.single i x))
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    have s₂ := f.map_vadd 0 (Pi.single i x)
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      i : ι
      x : φv i
      s₁ : Eq (f (Pi.single i x)) (g (Pi.single i x))
      s₂ : Eq (f (HVAdd.hVAdd (Pi.single i x) 0)) (HVAdd.hVAdd (f.linear (Pi.single  …
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    have s₃ := g.map_vadd 0 (Pi.single i x)
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      i : ι
      x : φv i
      s₁ : Eq (f (Pi.single i x)) (g (Pi.single i x))
      s₂ : Eq (f (HVAdd.hVAdd (Pi.single i x) 0)) (HVAdd.hVAdd (f.linear (Pi.single  …
      s₃ : Eq (g (HVAdd.hVAdd (Pi.single i x) 0)) (HVAdd.hVAdd (g.linear (Pi.single  …
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    rw [vadd_eq_add, add_zero] at s₂ s₃
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      i : ι
      x : φv i
      s₁ : Eq (f (Pi.single i x)) (g (Pi.single i x))
      s₂ : Eq (f (Pi.single i x)) (HVAdd.hVAdd (f.linear (Pi.single i x)) (f 0))
      s₃ : Eq (g (Pi.single i x)) (HVAdd.hVAdd (g.linear (Pi.single i x)) (g 0))
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    replace h₂ := h i 0
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      i : ι
      x : φv i
      s₁ : Eq (f (Pi.single i x)) (g (Pi.single i x))
      s₂ : Eq (f (Pi.single i x)) (HVAdd.hVAdd (f.linear (Pi.single i x)) (f 0))
      s₃ : Eq (g (Pi.single i x)) (HVAdd.hVAdd (g.linear (Pi.single i x)) (g 0))
      h₂ : Eq (f (Pi.single i 0)) (g (Pi.single i 0))
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    simp only [Pi.single_zero] at h₂
    /-
      case h₁.h
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      i : ι
      x : φv i
      s₁ : Eq (f (Pi.single i x)) (g (Pi.single i x))
      s₂ : Eq (f (Pi.single i x)) (HVAdd.hVAdd (f.linear (Pi.single i x)) (f 0))
      s₃ : Eq (g (Pi.single i x)) (HVAdd.hVAdd (g.linear (Pi.single i x)) (g 0))
      h₂ : Eq (f 0) (g 0)
      ⊢ Eq (f.linear (Pi.single i x)) (g.linear (Pi.single i x))
    -/
    rwa [s₂, s₃, h₂, vadd_right_cancel_iff] at s₁
    /-
      🎉 no goals
    -/
    /-
      case h₂
      k : Type u_2
      V2 : Type u_5
      P2 : Type u_6
      inst✝⁷ : Ring k
      inst✝⁶ : AddCommGroup V2
      inst✝⁵ : AddTorsor V2 P2
      inst✝⁴ : Module k V2
      ι : Type u_9
      φv : ι → Type u_10
      inst✝³ : (i : ι) → AddCommGroup (φv i)
      inst✝² : (i : ι) → Module k (φv i)
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      f g : AffineMap k ((i : ι) → φv i) P2
      h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
      h₂ : Eq (f 0) (g 0)
      ⊢ Eq (f ?p) (g ?p)
    -/
  · exact h₂
    /-
      🎉 no goals
    -/


/-- Two affine maps from a Pi-type of modules `(i : ι) → φv i` are equal if they are equal in their
  operation on `Pi.single` and `ι` is nonempty.  Analogous to `LinearMap.pi_ext`. See also
  `pi_ext_zero`, which instead `Nonempty ι` requires agreement at 0.-/
theorem pi_ext_nonempty [Nonempty ι] (h : ∀ i x, f (Pi.single i x) = g (Pi.single i x)) :
    f = g := by
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V2
    inst✝⁶ : AddTorsor V2 P2
    inst✝⁵ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝⁴ : (i : ι) → AddCommGroup (φv i)
    inst✝³ : (i : ι) → Module k (φv i)
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    inst✝ : Nonempty ι
    h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
    ⊢ Eq f g
  -/
  apply pi_ext_zero h
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V2
    inst✝⁶ : AddTorsor V2 P2
    inst✝⁵ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝⁴ : (i : ι) → AddCommGroup (φv i)
    inst✝³ : (i : ι) → Module k (φv i)
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    inst✝ : Nonempty ι
    h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
    ⊢ Eq (f 0) (g 0)
  -/
  inhabit ι
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V2
    inst✝⁶ : AddTorsor V2 P2
    inst✝⁵ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝⁴ : (i : ι) → AddCommGroup (φv i)
    inst✝³ : (i : ι) → Module k (φv i)
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    inst✝ : Nonempty ι
    h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
    inhabited_h : Inhabited ι
    ⊢ Eq (f 0) (g 0)
  -/
  rw [← Pi.single_zero default]
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V2
    inst✝⁶ : AddTorsor V2 P2
    inst✝⁵ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝⁴ : (i : ι) → AddCommGroup (φv i)
    inst✝³ : (i : ι) → Module k (φv i)
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    inst✝ : Nonempty ι
    h : ∀ (i : ι) (x : φv i), Eq (f (Pi.single i x)) (g (Pi.single i x))
    inhabited_h : Inhabited ι
    ⊢ Eq (f (Pi.single Inhabited.default 0)) (g (Pi.single Inhabited.default 0))
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- This is used as the ext lemma instead of `AffineMap.pi_ext_nonempty` for reasons explained in
note [partially-applied ext lemmas]. Analogous to `LinearMap.pi_ext'`-/
@[ext (iff := false)]
theorem pi_ext_nonempty' [Nonempty ι] (h : ∀ i, f.comp (LinearMap.single _ _ i).toAffineMap =
    g.comp (LinearMap.single _ _ i).toAffineMap) : f = g := by
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V2
    inst✝⁶ : AddTorsor V2 P2
    inst✝⁵ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝⁴ : (i : ι) → AddCommGroup (φv i)
    inst✝³ : (i : ι) → Module k (φv i)
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    inst✝ : Nonempty ι
    h : ∀ (i : ι), Eq (f.comp (LinearMap.single k φv i).toAffineMap) (g.comp (Line …
    ⊢ Eq f g
  -/
  refine pi_ext_nonempty fun i x => ?_
  /-
    k : Type u_2
    V2 : Type u_5
    P2 : Type u_6
    inst✝⁸ : Ring k
    inst✝⁷ : AddCommGroup V2
    inst✝⁶ : AddTorsor V2 P2
    inst✝⁵ : Module k V2
    ι : Type u_9
    φv : ι → Type u_10
    inst✝⁴ : (i : ι) → AddCommGroup (φv i)
    inst✝³ : (i : ι) → Module k (φv i)
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    f g : AffineMap k ((i : ι) → φv i) P2
    inst✝ : Nonempty ι
    h : ∀ (i : ι), Eq (f.comp (LinearMap.single k φv i).toAffineMap) (g.comp (Line …
    i : ι
    x : φv i
    ⊢ Eq (f (Pi.single i x)) (g (Pi.single i x))
  -/
  convert AffineMap.congr_fun (h i) x
  /-
    🎉 no goals
  -/


/-- `homothety c r` is the homothety (also known as dilation) about `c` with scale factor `r`. -/
def homothety (c : P1) (r : k) : P1 →ᵃ[k] P1 :=
  r • (id k P1 -ᵥ const k P1 c) +ᵥ const k P1 c


theorem homothety_def (c : P1) (r : k) :
    homothety c r = r • (id k P1 -ᵥ const k P1 c) +ᵥ const k P1 c :=
  rfl


theorem homothety_apply (c : P1) (r : k) (p : P1) : homothety c r p = r • (p -ᵥ c : V1) +ᵥ c :=
  rfl


theorem homothety_eq_lineMap (c : P1) (r : k) (p : P1) : homothety c r p = lineMap c p r :=
  rfl


@[simp]
theorem homothety_one (c : P1) : homothety c (1 : k) = id k P1 := by
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝³ : CommRing k
    inst✝² : AddCommGroup V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : Module k V1
    c : P1
    ⊢ Eq (AffineMap.homothety c 1) (AffineMap.id k P1)
  -/
  ext p
  /-
    case h
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝³ : CommRing k
    inst✝² : AddCommGroup V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : Module k V1
    c p : P1
    ⊢ Eq ((AffineMap.homothety c 1) p) ((AffineMap.id k P1) p)
  -/
  simp [homothety_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem homothety_apply_same (c : P1) (r : k) : homothety c r c = c :=
  lineMap_same_apply c r


theorem homothety_mul_apply (c : P1) (r₁ r₂ : k) (p : P1) :
    homothety c (r₁ * r₂) p = homothety c r₁ (homothety c r₂ p) := by
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝³ : CommRing k
    inst✝² : AddCommGroup V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : Module k V1
    c : P1
    r₁ r₂ : k
    p : P1
    ⊢ Eq ((AffineMap.homothety c (HMul.hMul r₁ r₂)) p) ((AffineMap.homothety c r₁) …
  -/
  simp only [homothety_apply, mul_smul, vadd_vsub]
  /-
    🎉 no goals
  -/


theorem homothety_mul (c : P1) (r₁ r₂ : k) :
    homothety c (r₁ * r₂) = (homothety c r₁).comp (homothety c r₂) :=
  ext <| homothety_mul_apply c r₁ r₂


@[simp]
theorem homothety_zero (c : P1) : homothety c (0 : k) = const k P1 c := by
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝³ : CommRing k
    inst✝² : AddCommGroup V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : Module k V1
    c : P1
    ⊢ Eq (AffineMap.homothety c 0) (AffineMap.const k P1 c)
  -/
  ext p
  /-
    case h
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝³ : CommRing k
    inst✝² : AddCommGroup V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : Module k V1
    c p : P1
    ⊢ Eq ((AffineMap.homothety c 0) p) ((AffineMap.const k P1 c) p)
  -/
  simp [homothety_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem homothety_add (c : P1) (r₁ r₂ : k) :
    homothety c (r₁ + r₂) = r₁ • (id k P1 -ᵥ const k P1 c) +ᵥ homothety c r₂ := by
  /-
    k : Type u_2
    V1 : Type u_3
    P1 : Type u_4
    inst✝³ : CommRing k
    inst✝² : AddCommGroup V1
    inst✝¹ : AddTorsor V1 P1
    inst✝ : Module k V1
    c : P1
    r₁ r₂ : k
    ⊢ Eq (AffineMap.homothety c (HAdd.hAdd r₁ r₂)) (HVAdd.hVAdd (HSMul.hSMul r₁ (V …
  -/
  simp only [homothety_def, add_smul, vadd_vadd]
  /-
    🎉 no goals
  -/


/-- `homothety` as a multiplicative monoid homomorphism. -/
def homothetyHom (c : P1) : k →* P1 →ᵃ[k] P1 where
  toFun := homothety c
  map_one' := homothety_one c
  map_mul' := homothety_mul c


@[simp]
theorem coe_homothetyHom (c : P1) : ⇑(homothetyHom c : k →* _) = homothety c :=
  rfl


/-- `homothety` as an affine map. -/
def homothetyAffine (c : P1) : k →ᵃ[k] P1 →ᵃ[k] P1 :=
  ⟨homothety c, (LinearMap.lsmul k _).flip (id k P1 -ᵥ const k P1 c),
    Function.swap (homothety_add c)⟩


@[simp]
theorem coe_homothetyAffine (c : P1) : ⇑(homothetyAffine c : k →ᵃ[k] _) = homothety c :=
  rfl


/-- Applying an affine map to an affine combination of two points yields an affine combination of
the images. -/
theorem Convex.combo_affine_apply {x y : E} {a b : 𝕜} {f : E →ᵃ[𝕜] F} (h : a + b = 1) :
    f (a • x + b • y) = a • f x + b • f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x y : E
    a b : 𝕜
    f : AffineMap 𝕜 E F
    h : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd (HSMul.hSM …
  -/
  simp only [Convex.combo_eq_smul_sub_add h, ← vsub_eq_sub]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : Ring 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    x y : E
    a b : 𝕜
    f : AffineMap 𝕜 E F
    h : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (f (HAdd.hAdd (HSMul.hSMul b (VSub.vsub y x)) x)) (HAdd.hAdd (HSMul.hSMul …
  -/
  exact f.apply_lineMap _ _ _
  /-
    🎉 no goals
  -/


