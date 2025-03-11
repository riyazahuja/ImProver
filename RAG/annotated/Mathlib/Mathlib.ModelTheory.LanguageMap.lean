/-- A language homomorphism maps the symbols of one language to symbols of another. -/
structure LHom where
  onFunction : ∀ ⦃n⦄, L.Functions n → L'.Functions n := by
    exact fun {n} => isEmptyElim
  onRelation : ∀ ⦃n⦄, L.Relations n → L'.Relations n :=by
    exact fun {n} => isEmptyElim


@[inherit_doc FirstOrder.Language.LHom]
infixl:10 " →ᴸ " => LHom

-- \^L

/-- Pulls a structure back along a language map. -/
def reduct (M : Type*) [L'.Structure M] : L.Structure M where
  funMap f xs := funMap (ϕ.onFunction f) xs
  RelMap r xs := RelMap (ϕ.onRelation r) xs


/-- The identity language homomorphism. -/
@[simps]
protected def id (L : Language) : L →ᴸ L :=
  ⟨fun _n => id, fun _n => id⟩


instance : Inhabited (L →ᴸ L) :=
  ⟨LHom.id L⟩


/-- The inclusion of the left factor into the sum of two languages. -/
@[simps]
protected def sumInl : L →ᴸ L.sum L' :=
  ⟨fun _n => Sum.inl, fun _n => Sum.inl⟩


/-- The inclusion of the right factor into the sum of two languages. -/
@[simps]
protected def sumInr : L' →ᴸ L.sum L' :=
  ⟨fun _n => Sum.inr, fun _n => Sum.inr⟩


/-- The inclusion of an empty language into any other language. -/
@[simps]
protected def ofIsEmpty [L.IsAlgebraic] [L.IsRelational] : L →ᴸ L' where


@[ext]
protected theorem funext {F G : L →ᴸ L'} (h_fun : F.onFunction = G.onFunction)
    (h_rel : F.onRelation = G.onRelation) : F = G := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    F G : L.LHom L'
    h_fun : Eq F.onFunction G.onFunction
    h_rel : Eq F.onRelation G.onRelation
    ⊢ Eq F G
  -/
  cases' F with Ff Fr
  /-
    case mk
    L : FirstOrder.Language
    L' : FirstOrder.Language
    G : L.LHom L'
    Ff : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    Fr : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    h_fun : Eq { onFunction := Ff, onRelation := Fr }.onFunction G.onFunction
    h_rel : Eq { onFunction := Ff, onRelation := Fr }.onRelation G.onRelation
    ⊢ Eq { onFunction := Ff, onRelation := Fr } G
  -/
  cases' G with Gf Gr
  /-
    case mk.mk
    L : FirstOrder.Language
    L' : FirstOrder.Language
    Ff : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    Fr : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    Gf : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    Gr : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    h_fun : Eq { onFunction := Ff, onRelation := Fr }.onFunction { onFunction := G …
    h_rel : Eq { onFunction := Ff, onRelation := Fr }.onRelation { onFunction := G …
    ⊢ Eq { onFunction := Ff, onRelation := Fr } { onFunction := Gf, onRelation :=  …
  -/
  simp only [mk.injEq]
  /-
    case mk.mk
    L : FirstOrder.Language
    L' : FirstOrder.Language
    Ff : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    Fr : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    Gf : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    Gr : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    h_fun : Eq { onFunction := Ff, onRelation := Fr }.onFunction { onFunction := G …
    h_rel : Eq { onFunction := Ff, onRelation := Fr }.onRelation { onFunction := G …
    ⊢ And (Eq Ff Gf) (Eq Fr Gr)
  -/
  exact And.intro h_fun h_rel
  /-
    🎉 no goals
  -/


instance [L.IsAlgebraic] [L.IsRelational] : Unique (L →ᴸ L') :=
  ⟨⟨LHom.ofIsEmpty L L'⟩, fun _ => LHom.funext (Subsingleton.elim _ _) (Subsingleton.elim _ _)⟩


/-- The composition of two language homomorphisms. -/
@[simps]
def comp (g : L' →ᴸ L'') (f : L →ᴸ L') : L →ᴸ L'' :=
  ⟨fun _n F => g.1 (f.1 F), fun _ R => g.2 (f.2 R)⟩

-- Porting note: added ᴸ to avoid clash with function composition

@[inherit_doc]
local infixl:60 " ∘ᴸ " => LHom.comp


@[simp]
theorem id_comp (F : L →ᴸ L') : LHom.id L' ∘ᴸ F = F := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    F : L.LHom L'
    ⊢ Eq ((FirstOrder.Language.LHom.id L').comp F) F
  -/
  cases F
  /-
    case mk
    L : FirstOrder.Language
    L' : FirstOrder.Language
    onFunction✝ : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    onRelation✝ : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    ⊢ Eq ((FirstOrder.Language.LHom.id L').comp { onFunction := onFunction✝, onRel …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_id (F : L →ᴸ L') : F ∘ᴸ LHom.id L = F := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    F : L.LHom L'
    ⊢ Eq (F.comp (FirstOrder.Language.LHom.id L)) F
  -/
  cases F
  /-
    case mk
    L : FirstOrder.Language
    L' : FirstOrder.Language
    onFunction✝ : ⦃n : Nat⦄ → L.Functions n → L'.Functions n
    onRelation✝ : ⦃n : Nat⦄ → L.Relations n → L'.Relations n
    ⊢ Eq ({ onFunction := onFunction✝, onRelation := onRelation✝ }.comp (FirstOrde …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comp_assoc {L3 : Language} (F : L'' →ᴸ L3) (G : L' →ᴸ L'') (H : L →ᴸ L') :
    F ∘ᴸ G ∘ᴸ H = F ∘ᴸ (G ∘ᴸ H) :=
  rfl


/-- A language map defined on two factors of a sum. -/
@[simps]
protected def sumElim : L.sum L'' →ᴸ L' where
  onFunction _n := Sum.elim (fun f => ϕ.onFunction f) fun f => ψ.onFunction f
  onRelation _n := Sum.elim (fun f => ϕ.onRelation f) fun f => ψ.onRelation f


theorem sumElim_comp_inl (ψ : L'' →ᴸ L') : ϕ.sumElim ψ ∘ᴸ LHom.sumInl = ϕ :=
  LHom.funext (funext fun _ => rfl) (funext fun _ => rfl)


theorem sumElim_comp_inr (ψ : L'' →ᴸ L') : ϕ.sumElim ψ ∘ᴸ LHom.sumInr = ψ :=
  LHom.funext (funext fun _ => rfl) (funext fun _ => rfl)


theorem sumElim_inl_inr : LHom.sumInl.sumElim LHom.sumInr = LHom.id (L.sum L') :=
  LHom.funext (funext fun _ => Sum.elim_inl_inr) (funext fun _ => Sum.elim_inl_inr)


theorem comp_sumElim {L3 : Language} (θ : L' →ᴸ L3) :
    θ ∘ᴸ ϕ.sumElim ψ = (θ ∘ᴸ ϕ).sumElim (θ ∘ᴸ ψ) :=
  LHom.funext (funext fun _n => Sum.comp_elim _ _ _) (funext fun _n => Sum.comp_elim _ _ _)


/-- The map between two sum-languages induced by maps on the two factors. -/
@[simps]
def sumMap : L.sum L₁ →ᴸ L'.sum L₂ where
  onFunction _n := Sum.map (fun f => ϕ.onFunction f) fun f => ψ.onFunction f
  onRelation _n := Sum.map (fun f => ϕ.onRelation f) fun f => ψ.onRelation f


@[simp]
theorem sumMap_comp_inl : ϕ.sumMap ψ ∘ᴸ LHom.sumInl = LHom.sumInl ∘ᴸ ϕ :=
  LHom.funext (funext fun _ => rfl) (funext fun _ => rfl)


@[simp]
theorem sumMap_comp_inr : ϕ.sumMap ψ ∘ᴸ LHom.sumInr = LHom.sumInr ∘ᴸ ψ :=
  LHom.funext (funext fun _ => rfl) (funext fun _ => rfl)


/-- A language homomorphism is injective when all the maps between symbol types are. -/
protected structure Injective : Prop where
  onFunction {n} : Function.Injective fun f : L.Functions n => onFunction ϕ f
  onRelation {n} : Function.Injective fun R : L.Relations n => onRelation ϕ R


/-- Pulls an `L`-structure along a language map `ϕ : L →ᴸ L'`, and then expands it
  to an `L'`-structure arbitrarily. -/
noncomputable def defaultExpansion (ϕ : L →ᴸ L')
    [∀ (n) (f : L'.Functions n), Decidable (f ∈ Set.range fun f : L.Functions n => onFunction ϕ f)]
    [∀ (n) (r : L'.Relations n), Decidable (r ∈ Set.range fun r : L.Relations n => onRelation ϕ r)]
    (M : Type*) [Inhabited M] [L.Structure M] : L'.Structure M where
  funMap {n} f xs :=
    if h' : f ∈ Set.range fun f : L.Functions n => onFunction ϕ f then funMap h'.choose xs
    else default
  RelMap {n} r xs :=
    if h' : r ∈ Set.range fun r : L.Relations n => onRelation ϕ r then RelMap h'.choose xs
    else default


/-- A language homomorphism is an expansion on a structure if it commutes with the interpretation of
all symbols on that structure. -/
class IsExpansionOn (M : Type*) [L.Structure M] [L'.Structure M] : Prop where
  map_onFunction :
    ∀ {n} (f : L.Functions n) (x : Fin n → M), funMap (ϕ.onFunction f) x = funMap f x := by
      exact fun {n} => isEmptyElim
  map_onRelation :
    ∀ {n} (R : L.Relations n) (x : Fin n → M), RelMap (ϕ.onRelation R) x = RelMap R x := by
      exact fun {n} => isEmptyElim


@[simp]
theorem map_onFunction {M : Type*} [L.Structure M] [L'.Structure M] [ϕ.IsExpansionOn M] {n}
    (f : L.Functions n) (x : Fin n → M) : funMap (ϕ.onFunction f) x = funMap f x :=
  IsExpansionOn.map_onFunction f x


@[simp]
theorem map_onRelation {M : Type*} [L.Structure M] [L'.Structure M] [ϕ.IsExpansionOn M] {n}
    (R : L.Relations n) (x : Fin n → M) : RelMap (ϕ.onRelation R) x = RelMap R x :=
  IsExpansionOn.map_onRelation R x


instance id_isExpansionOn (M : Type*) [L.Structure M] : IsExpansionOn (LHom.id L) M :=
  ⟨fun _ _ => rfl, fun _ _ => rfl⟩


instance ofIsEmpty_isExpansionOn (M : Type*) [L.Structure M] [L'.Structure M] [L.IsAlgebraic]
    [L.IsRelational] : IsExpansionOn (LHom.ofIsEmpty L L') M where


instance sumElim_isExpansionOn {L'' : Language} (ψ : L'' →ᴸ L') (M : Type*) [L.Structure M]
    [L'.Structure M] [L''.Structure M] [ϕ.IsExpansionOn M] [ψ.IsExpansionOn M] :
    (ϕ.sumElim ψ).IsExpansionOn M :=
                                /-
                                  L : FirstOrder.Language
                                  L' : FirstOrder.Language
                                  M✝ : Type w
                                  inst✝⁵ : L.Structure M✝
                                  ϕ : L.LHom L'
                                  L''✝ : FirstOrder.Language
                                  L'' : FirstOrder.Language
                                  ψ : L''.LHom L'
                                  M : Type u_1
                                  inst✝⁴ : L.Structure M
                                  inst✝³ : L'.Structure M
                                  inst✝² : L''.Structure M
                                  inst✝¹ : ϕ.IsExpansionOn M
                                  inst✝ : ψ.IsExpansionOn M
                                  n✝ : Nat
                                  f : (L.sum L'').Functions n✝
                                  x✝ : Fin n✝ → M
                                  ⊢ ∀ (val : L.Functions n✝), Eq (FirstOrder.Language.Structure.funMap ((ϕ.sumEl …
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
  ⟨fun f _ => Sum.casesOn f (by simp) (by simp), fun R _ => Sum.casesOn R (by simp) (by simp)⟩
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


instance sumMap_isExpansionOn {L₁ L₂ : Language} (ψ : L₁ →ᴸ L₂) (M : Type*) [L.Structure M]
    [L'.Structure M] [L₁.Structure M] [L₂.Structure M] [ϕ.IsExpansionOn M] [ψ.IsExpansionOn M] :
    (ϕ.sumMap ψ).IsExpansionOn M :=
                                /-
                                  L : FirstOrder.Language
                                  L' : FirstOrder.Language
                                  M✝ : Type w
                                  inst✝⁶ : L.Structure M✝
                                  ϕ : L.LHom L'
                                  L'' : FirstOrder.Language
                                  L₁ : FirstOrder.Language
                                  L₂ : FirstOrder.Language
                                  ψ : L₁.LHom L₂
                                  M : Type u_1
                                  inst✝⁵ : L.Structure M
                                  inst✝⁴ : L'.Structure M
                                  inst✝³ : L₁.Structure M
                                  inst✝² : L₂.Structure M
                                  inst✝¹ : ϕ.IsExpansionOn M
                                  inst✝ : ψ.IsExpansionOn M
                                  n✝ : Nat
                                  f : (L.sum L₁).Functions n✝
                                  x✝ : Fin n✝ → M
                                  ⊢ ∀ (val : L.Functions n✝), Eq (FirstOrder.Language.Structure.funMap ((ϕ.sumMa …
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
  ⟨fun f _ => Sum.casesOn f (by simp) (by simp), fun R _ => Sum.casesOn R (by simp) (by simp)⟩
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


instance sumInl_isExpansionOn (M : Type*) [L.Structure M] [L'.Structure M] :
    (LHom.sumInl : L →ᴸ L.sum L').IsExpansionOn M :=
  ⟨fun _f _ => rfl, fun _R _ => rfl⟩


instance sumInr_isExpansionOn (M : Type*) [L.Structure M] [L'.Structure M] :
    (LHom.sumInr : L' →ᴸ L.sum L').IsExpansionOn M :=
  ⟨fun _f _ => rfl, fun _R _ => rfl⟩


@[simp]
theorem funMap_sumInl [(L.sum L').Structure M] [(LHom.sumInl : L →ᴸ L.sum L').IsExpansionOn M] {n}
    {f : L.Functions n} {x : Fin n → M} : @funMap (L.sum L') M _ n (Sum.inl f) x = funMap f x :=
  (LHom.sumInl : L →ᴸ L.sum L').map_onFunction f x


@[simp]
theorem funMap_sumInr [(L'.sum L).Structure M] [(LHom.sumInr : L →ᴸ L'.sum L).IsExpansionOn M] {n}
    {f : L.Functions n} {x : Fin n → M} : @funMap (L'.sum L) M _ n (Sum.inr f) x = funMap f x :=
  (LHom.sumInr : L →ᴸ L'.sum L).map_onFunction f x


theorem sumInl_injective : (LHom.sumInl : L →ᴸ L.sum L').Injective :=
  ⟨fun h => Sum.inl_injective h, fun h => Sum.inl_injective h⟩


theorem sumInr_injective : (LHom.sumInr : L' →ᴸ L.sum L').Injective :=
  ⟨fun h => Sum.inr_injective h, fun h => Sum.inr_injective h⟩


instance (priority := 100) isExpansionOn_reduct (ϕ : L →ᴸ L') (M : Type*) [L'.Structure M] :
    @IsExpansionOn L L' ϕ M (ϕ.reduct M) _ :=
  letI := ϕ.reduct M
  ⟨fun _f _ => rfl, fun _R _ => rfl⟩


theorem Injective.isExpansionOn_default {ϕ : L →ᴸ L'}
    [∀ (n) (f : L'.Functions n), Decidable (f ∈ Set.range fun f : L.Functions n => ϕ.onFunction f)]
    [∀ (n) (r : L'.Relations n), Decidable (r ∈ Set.range fun r : L.Relations n => ϕ.onRelation r)]
    (h : ϕ.Injective) (M : Type*) [Inhabited M] [L.Structure M] :
    @IsExpansionOn L L' ϕ M _ (ϕ.defaultExpansion M) := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    ϕ : L.LHom L'
    inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
    inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
    h : ϕ.Injective
    M : Type u_1
    inst✝¹ : Inhabited M
    inst✝ : L.Structure M
    ⊢ ϕ.IsExpansionOn M
  -/
  letI := ϕ.defaultExpansion M
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    ϕ : L.LHom L'
    inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
    inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
    h : ϕ.Injective
    M : Type u_1
    inst✝¹ : Inhabited M
    inst✝ : L.Structure M
    this : L'.Structure M := ϕ.defaultExpansion M
    ⊢ ϕ.IsExpansionOn M
  -/
  refine ⟨fun {n} f xs => ?_, fun {n} r xs => ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      L' : FirstOrder.Language
      ϕ : L.LHom L'
      inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
      inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
      h : ϕ.Injective
      M : Type u_1
      inst✝¹ : Inhabited M
      inst✝ : L.Structure M
      this : L'.Structure M := ϕ.defaultExpansion M
      n : Nat
      f : L.Functions n
      xs : Fin n → M
      ⊢ Eq (FirstOrder.Language.Structure.funMap (ϕ.onFunction f) xs) (FirstOrder.La …
    -/
  · have hf : ϕ.onFunction f ∈ Set.range fun f : L.Functions n => ϕ.onFunction f := ⟨f, rfl⟩
    /-
      case refine_1
      L : FirstOrder.Language
      L' : FirstOrder.Language
      ϕ : L.LHom L'
      inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
      inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
      h : ϕ.Injective
      M : Type u_1
      inst✝¹ : Inhabited M
      inst✝ : L.Structure M
      this : L'.Structure M := ϕ.defaultExpansion M
      n : Nat
      f : L.Functions n
      xs : Fin n → M
      hf : Membership.mem (Set.range fun f => ϕ.onFunction f) (ϕ.onFunction f)
      ⊢ Eq (FirstOrder.Language.Structure.funMap (ϕ.onFunction f) xs) (FirstOrder.La …
    -/
    refine (dif_pos hf).trans ?_
    /-
      case refine_1
      L : FirstOrder.Language
      L' : FirstOrder.Language
      ϕ : L.LHom L'
      inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
      inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
      h : ϕ.Injective
      M : Type u_1
      inst✝¹ : Inhabited M
      inst✝ : L.Structure M
      this : L'.Structure M := ϕ.defaultExpansion M
      n : Nat
      f : L.Functions n
      xs : Fin n → M
      hf : Membership.mem (Set.range fun f => ϕ.onFunction f) (ϕ.onFunction f)
      ⊢ Eq (FirstOrder.Language.Structure.funMap (Exists.choose hf) xs) (FirstOrder. …
    -/
    rw [h.onFunction hf.choose_spec]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      L' : FirstOrder.Language
      ϕ : L.LHom L'
      inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
      inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
      h : ϕ.Injective
      M : Type u_1
      inst✝¹ : Inhabited M
      inst✝ : L.Structure M
      this : L'.Structure M := ϕ.defaultExpansion M
      n : Nat
      r : L.Relations n
      xs : Fin n → M
      ⊢ Eq (FirstOrder.Language.Structure.RelMap (ϕ.onRelation r) xs) (FirstOrder.La …
    -/
  · have hr : ϕ.onRelation r ∈ Set.range fun r : L.Relations n => ϕ.onRelation r := ⟨r, rfl⟩
    /-
      case refine_2
      L : FirstOrder.Language
      L' : FirstOrder.Language
      ϕ : L.LHom L'
      inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
      inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
      h : ϕ.Injective
      M : Type u_1
      inst✝¹ : Inhabited M
      inst✝ : L.Structure M
      this : L'.Structure M := ϕ.defaultExpansion M
      n : Nat
      r : L.Relations n
      xs : Fin n → M
      hr : Membership.mem (Set.range fun r => ϕ.onRelation r) (ϕ.onRelation r)
      ⊢ Eq (FirstOrder.Language.Structure.RelMap (ϕ.onRelation r) xs) (FirstOrder.La …
    -/
    refine (dif_pos hr).trans ?_
    /-
      case refine_2
      L : FirstOrder.Language
      L' : FirstOrder.Language
      ϕ : L.LHom L'
      inst✝³ : (n : Nat) → (f : L'.Functions n) → Decidable (Membership.mem (Set.ran …
      inst✝² : (n : Nat) → (r : L'.Relations n) → Decidable (Membership.mem (Set.ran …
      h : ϕ.Injective
      M : Type u_1
      inst✝¹ : Inhabited M
      inst✝ : L.Structure M
      this : L'.Structure M := ϕ.defaultExpansion M
      n : Nat
      r : L.Relations n
      xs : Fin n → M
      hr : Membership.mem (Set.range fun r => ϕ.onRelation r) (ϕ.onRelation r)
      ⊢ Eq (FirstOrder.Language.Structure.RelMap (Exists.choose hr) xs) (FirstOrder. …
    -/
    rw [h.onRelation hr.choose_spec]
    /-
      🎉 no goals
    -/


/-- A language equivalence maps the symbols of one language to symbols of another bijectively. -/
structure LEquiv (L L' : Language) where
  toLHom : L →ᴸ L'
  invLHom : L' →ᴸ L
  left_inv : invLHom.comp toLHom = LHom.id L
  right_inv : toLHom.comp invLHom = LHom.id L'


@[inherit_doc] infixl:10 " ≃ᴸ " => LEquiv

-- \^L

/-- The identity equivalence from a first-order language to itself. -/
@[simps]
protected def refl : L ≃ᴸ L :=
  ⟨LHom.id L, LHom.id L, LHom.comp_id _, LHom.comp_id _⟩


instance : Inhabited (L ≃ᴸ L) :=
  ⟨LEquiv.refl L⟩


/-- The inverse of an equivalence of first-order languages. -/
@[simps]
protected def symm : L' ≃ᴸ L :=
  ⟨e.invLHom, e.toLHom, e.right_inv, e.left_inv⟩


/-- The composition of equivalences of first-order languages. -/
@[simps, trans]
protected def trans (e : L ≃ᴸ L') (e' : L' ≃ᴸ L'') : L ≃ᴸ L'' :=
  ⟨e'.toLHom.comp e.toLHom, e.invLHom.comp e'.invLHom, by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      L'' : FirstOrder.Language
      e'✝ : L'.LEquiv L''
      e✝ e : L.LEquiv L'
      e' : L'.LEquiv L''
      ⊢ Eq ((e.invLHom.comp e'.invLHom).comp (e'.toLHom.comp e.toLHom)) (FirstOrder. …
    -/
    rw [LHom.comp_assoc, ← LHom.comp_assoc e'.invLHom, e'.left_inv, LHom.id_comp, e.left_inv], by
    /-
      🎉 no goals
    -/
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      L'' : FirstOrder.Language
      e'✝ : L'.LEquiv L''
      e✝ e : L.LEquiv L'
      e' : L'.LEquiv L''
      ⊢ Eq ((e'.toLHom.comp e.toLHom).comp (e.invLHom.comp e'.invLHom)) (FirstOrder. …
    -/
    rw [LHom.comp_assoc, ← LHom.comp_assoc e.toLHom, e.right_inv, LHom.id_comp, e'.right_inv]⟩
    /-
      🎉 no goals
    -/


/-- The type of functions for a language consisting only of constant symbols. -/
@[simp]
def constantsOnFunc : ℕ → Type u'
  | 0 => α
  | (_ + 1) => PEmpty


/-- A language with constants indexed by a type. -/
@[simps]
def constantsOn : Language.{u', 0} := ⟨constantsOnFunc α, fun _ => Empty⟩


theorem constantsOn_constants : (constantsOn α).Constants = α :=
  rfl


instance isAlgebraic_constantsOn : IsAlgebraic (constantsOn α) := by
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    ⊢ (FirstOrder.Language.constantsOn α).IsAlgebraic
  -/
  unfold constantsOn
  /-
    L : FirstOrder.Language
    L' : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    α : Type u'
    ⊢ { Functions := FirstOrder.Language.constantsOnFunc α, Relations := fun x =>  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isEmpty_functions_constantsOn_succ {n : ℕ} : IsEmpty ((constantsOn α).Functions (n + 1)) :=
  inferInstanceAs (IsEmpty PEmpty)


instance isRelational_constantsOn [_ie : IsEmpty α] : IsRelational (constantsOn α) :=
  fun n => Nat.casesOn n _ie inferInstance


theorem card_constantsOn : (constantsOn α).card = #α := by
  /-
    α : Type u'
    ⊢ Eq (FirstOrder.Language.constantsOn α).card (Cardinal.mk α)
  -/
  simp [card_eq_card_functions_add_card_relations, sum_nat_eq_add_sum_succ]
  /-
    🎉 no goals
  -/


/-- Gives a `constantsOn α` structure to a type by assigning each constant a value. -/
def constantsOn.structure (f : α → M) : (constantsOn α).Structure M where
  funMap := fun {n} c _ =>
    match n, c with
    | 0, c => f c


/-- A map between index types induces a map between constant languages. -/
def LHom.constantsOnMap (f : α → β) : constantsOn α →ᴸ constantsOn β where
  onFunction := fun {n} c =>
    match n, c with
    | 0, c => f c


theorem constantsOnMap_isExpansionOn {f : α → β} {fα : α → M} {fβ : β → M} (h : fβ ∘ f = fα) :
    @LHom.IsExpansionOn _ _ (LHom.constantsOnMap f) M (constantsOn.structure fα)
      (constantsOn.structure fβ) := by
  /-
    M : Type w
    α : Type u'
    β : Type v'
    f : α → β
    fα : α → M
    fβ : β → M
    h : Eq (Function.comp fβ f) fα
    ⊢ (FirstOrder.Language.LHom.constantsOnMap f).IsExpansionOn M
  -/
  letI := constantsOn.structure fα
  /-
    M : Type w
    α : Type u'
    β : Type v'
    f : α → β
    fα : α → M
    fβ : β → M
    h : Eq (Function.comp fβ f) fα
    this : (FirstOrder.Language.constantsOn α).Structure M := FirstOrder.Language. …
    ⊢ (FirstOrder.Language.LHom.constantsOnMap f).IsExpansionOn M
  -/
  letI := constantsOn.structure fβ
  exact
    ⟨fun {n} => Nat.casesOn n (fun F _x => (congr_fun h F : _)) fun n F => isEmptyElim F, fun R =>
      isEmptyElim R⟩


/-- Extends a language with a constant for each element of a parameter set in `M`. -/
def withConstants : Language.{max u w', v} :=
  L.sum (constantsOn α)


@[inherit_doc FirstOrder.Language.withConstants]
scoped[FirstOrder] notation:95 L "[[" α "]]" => Language.withConstants L α


@[simp]
theorem card_withConstants :
    L[[α]].card = Cardinal.lift.{w'} L.card + Cardinal.lift.{max u v} #α := by
  /-
    L : FirstOrder.Language
    α : Type w'
    ⊢ Eq (L.withConstants α).card (HAdd.hAdd (Cardinal.lift.{w', max u v} L.card)  …
  -/
  rw [withConstants, card_sum, card_constantsOn]
  /-
    🎉 no goals
  -/


/-- The language map adding constants. -/
@[simps!] -- Porting note: add `!` to `simps`
def lhomWithConstants : L →ᴸ L[[α]] :=
  LHom.sumInl


theorem lhomWithConstants_injective : (L.lhomWithConstants α).Injective :=
  LHom.sumInl_injective


/-- The constant symbol indexed by a particular element. -/
protected def con (a : α) : L[[α]].Constants :=
  Sum.inr a


/-- Adds constants to a language map. -/
def LHom.addConstants {L' : Language} (φ : L →ᴸ L') : L[[α]] →ᴸ L'[[α]] :=
  φ.sumMap (LHom.id _)


instance paramsStructure (A : Set α) : (constantsOn A).Structure α :=
  constantsOn.structure (↑)


/-- The language map removing an empty constant set. -/
@[simps]
def LEquiv.addEmptyConstants [ie : IsEmpty α] : L ≃ᴸ L[[α]] where
  toLHom := lhomWithConstants L α
  invLHom := LHom.sumElim (LHom.id L) (LHom.ofIsEmpty (constantsOn α) L)
                 /-
                   L : FirstOrder.Language
                   L' : FirstOrder.Language
                   M : Type w
                   inst✝ : L.Structure M
                   α : Type w'
                   ie : IsEmpty α
                   ⊢ Eq (((FirstOrder.Language.LHom.id L).sumElim (FirstOrder.Language.LHom.ofIsE …
                 -/
  left_inv := by rw [lhomWithConstants, LHom.sumElim_comp_inl]
                 /-
                   🎉 no goals
                 -/
  right_inv := by
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type w'
      ie : IsEmpty α
      ⊢ Eq ((L.lhomWithConstants α).comp ((FirstOrder.Language.LHom.id L).sumElim (F …
    -/
    simp only [LHom.comp_sumElim, lhomWithConstants, LHom.comp_id]
    /-
      L : FirstOrder.Language
      L' : FirstOrder.Language
      M : Type w
      inst✝ : L.Structure M
      α : Type w'
      ie : IsEmpty α
      ⊢ Eq (FirstOrder.Language.LHom.sumInl.sumElim (FirstOrder.Language.LHom.sumInl …
    -/
    exact _root_.trans (congr rfl (Subsingleton.elim _ _)) LHom.sumElim_inl_inr
    /-
      🎉 no goals
    -/


@[simp]
theorem withConstants_funMap_sum_inl [L[[α]].Structure M] [(lhomWithConstants L α).IsExpansionOn M]
    {n} {f : L.Functions n} {x : Fin n → M} : @funMap (L[[α]]) M _ n (Sum.inl f) x = funMap f x :=
  (lhomWithConstants L α).map_onFunction f x


@[simp]
theorem withConstants_relMap_sum_inl [L[[α]].Structure M] [(lhomWithConstants L α).IsExpansionOn M]
    {n} {R : L.Relations n} {x : Fin n → M} : @RelMap (L[[α]]) M _ n (Sum.inl R) x = RelMap R x :=
  (lhomWithConstants L α).map_onRelation R x


/-- The language map extending the constant set. -/
def lhomWithConstantsMap (f : α → β) : L[[α]] →ᴸ L[[β]] :=
  LHom.sumMap (LHom.id L) (LHom.constantsOnMap f)


@[simp]
theorem LHom.map_constants_comp_sumInl {f : α → β} :
                                                                              /-
                                                                                L : FirstOrder.Language
                                                                                α : Type w'
                                                                                β : Type u_1
                                                                                f : α → β
                                                                                ⊢ Eq ((L.lhomWithConstantsMap f).comp FirstOrder.Language.LHom.sumInl) (L.lhom …
                                                                              -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    (L.lhomWithConstantsMap f).comp LHom.sumInl = L.lhomWithConstants β := by ext <;> rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


instance constantsOnSelfStructure : (constantsOn M).Structure M :=
  constantsOn.structure id


instance withConstantsSelfStructure : L[[M]].Structure M :=
  Language.sumStructure _ _ M


instance withConstants_self_expansion : (lhomWithConstants L M).IsExpansionOn M :=
  ⟨fun _ _ => rfl, fun _ _ => rfl⟩


instance withConstantsStructure : L[[α]].Structure M :=
  Language.sumStructure _ _ _


instance withConstants_expansion : (L.lhomWithConstants α).IsExpansionOn M :=
  ⟨fun _ _ => rfl, fun _ _ => rfl⟩


instance addEmptyConstants_is_expansion_on' :
    (LEquiv.addEmptyConstants L (∅ : Set M)).toLHom.IsExpansionOn M :=
  L.withConstants_expansion _


instance addEmptyConstants_symm_isExpansionOn :
    (LEquiv.addEmptyConstants L (∅ : Set M)).symm.toLHom.IsExpansionOn M :=
  LHom.sumElim_isExpansionOn _ _ _


instance addConstants_expansion {L' : Language} [L'.Structure M] (φ : L →ᴸ L') [φ.IsExpansionOn M] :
    (φ.addConstants α).IsExpansionOn M :=
  LHom.sumMap_isExpansionOn _ _ M


@[simp]
theorem withConstants_funMap_sum_inr {a : α} {x : Fin 0 → M} :
    @funMap (L[[α]]) M _ 0 (Sum.inr a : L[[α]].Functions 0) x = L.con a := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u_1
    inst✝ : (FirstOrder.Language.constantsOn α).Structure M
    a : α
    x : Fin 0 → M
    ⊢ Eq (FirstOrder.Language.Structure.funMap (Sum.inr a) x) ↑(L.con a)
  -/
  rw [Unique.eq_default x]
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝¹ : L.Structure M
    α : Type u_1
    inst✝ : (FirstOrder.Language.constantsOn α).Structure M
    a : α
    x : Fin 0 → M
    ⊢ Eq (FirstOrder.Language.Structure.funMap (Sum.inr a) Inhabited.default) ↑(L. …
  -/
  exact (LHom.sumInr : constantsOn α →ᴸ L.sum _).map_onFunction _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_con {a : A} : (L.con a : M) = a :=
  rfl


instance constantsOnMap_inclusion_isExpansionOn :
    (LHom.constantsOnMap (Set.inclusion h)).IsExpansionOn M :=
  constantsOnMap_isExpansionOn rfl


instance map_constants_inclusion_isExpansionOn :
    (L.lhomWithConstantsMap (Set.inclusion h)).IsExpansionOn M :=
  LHom.sumMap_isExpansionOn _ _ _


