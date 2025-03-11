/-- The type of clopen upper sets of a topological space. -/
structure ClopenUpperSet (α : Type*) [TopologicalSpace α] [LE α] extends Clopens α where
  upper' : IsUpperSet carrier


instance : SetLike (ClopenUpperSet α) α where
  coe s := s.carrier
  coe_injective' s t h := by
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : LE α
      s t : ClopenUpperSet α
      h : Eq ((fun s => s.carrier) s) ((fun s => s.carrier) t)
      ⊢ Eq s t
    -/
    obtain ⟨⟨_, _⟩, _⟩ := s
    /-
      case mk.mk
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : LE α
      t : ClopenUpperSet α
      carrier✝ : Set α
      isClopen'✝ : IsClopen carrier✝
      upper'✝ : IsUpperSet { carrier := carrier✝, isClopen' := isClopen'✝ }.carrier
      h : Eq ((fun s => s.carrier) { carrier := carrier✝, isClopen' := isClopen'✝, u …
      ⊢ Eq { carrier := carrier✝, isClopen' := isClopen'✝, upper' := upper'✝ } t
    -/
    obtain ⟨⟨_, _⟩, _⟩ := t
    /-
      case mk.mk.mk.mk
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : LE α
      carrier✝¹ : Set α
      isClopen'✝¹ : IsClopen carrier✝¹
      upper'✝¹ : IsUpperSet { carrier := carrier✝¹, isClopen' := isClopen'✝¹ }.carrier
      carrier✝ : Set α
      isClopen'✝ : IsClopen carrier✝
      upper'✝ : IsUpperSet { carrier := carrier✝, isClopen' := isClopen'✝ }.carrier
      h : Eq ((fun s => s.carrier) { carrier := carrier✝¹, isClopen' := isClopen'✝¹, …
      ⊢ Eq { carrier := carrier✝¹, isClopen' := isClopen'✝¹, upper' := upper'✝¹ } {  …
    -/
    congr
    /-
      🎉 no goals
    -/


/-- See Note [custom simps projection]. -/
def Simps.coe (s : ClopenUpperSet α) : Set α := s


theorem upper (s : ClopenUpperSet α) : IsUpperSet (s : Set α) :=
  s.upper'


theorem isClopen (s : ClopenUpperSet α) : IsClopen (s : Set α) :=
  s.isClopen'


/-- Reinterpret an upper clopen as an upper set. -/
@[simps]
def toUpperSet (s : ClopenUpperSet α) : UpperSet α :=
  ⟨s, s.upper⟩


@[ext]
protected theorem ext {s t : ClopenUpperSet α} (h : (s : Set α) = t) : s = t :=
  SetLike.ext' h


@[simp]
theorem coe_mk (s : Clopens α) (h) : (mk s h : Set α) = s :=
  rfl


instance : Max (ClopenUpperSet α) :=
  ⟨fun s t => ⟨s.toClopens ⊔ t.toClopens, s.upper.union t.upper⟩⟩


instance : Min (ClopenUpperSet α) :=
  ⟨fun s t => ⟨s.toClopens ⊓ t.toClopens, s.upper.inter t.upper⟩⟩


instance : Top (ClopenUpperSet α) :=
  ⟨⟨⊤, isUpperSet_univ⟩⟩


instance : Bot (ClopenUpperSet α) :=
  ⟨⟨⊥, isUpperSet_empty⟩⟩


instance : Lattice (ClopenUpperSet α) :=
  SetLike.coe_injective.lattice _ (fun _ _ => rfl) fun _ _ => rfl


instance : BoundedOrder (ClopenUpperSet α) :=
  BoundedOrder.lift ((↑) : _ → Set α) (fun _ _ => id) rfl rfl


@[simp]
theorem coe_sup (s t : ClopenUpperSet α) : (↑(s ⊔ t) : Set α) = ↑s ∪ ↑t :=
  rfl


@[simp]
theorem coe_inf (s t : ClopenUpperSet α) : (↑(s ⊓ t) : Set α) = ↑s ∩ ↑t :=
  rfl


@[simp]
theorem coe_top : (↑(⊤ : ClopenUpperSet α) : Set α) = univ :=
  rfl


@[simp]
theorem coe_bot : (↑(⊥ : ClopenUpperSet α) : Set α) = ∅ :=
  rfl


instance : Inhabited (ClopenUpperSet α) :=
  ⟨⊥⟩


